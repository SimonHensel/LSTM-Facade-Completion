import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell, LSTMStateTuple
from tensorflow.contrib import rnn as rnn_cell
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import _linear
from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.contrib import grid_rnn
import tensorflow.contrib.slim as slim
from resnet import DeepLabResNetModel
from to_hilbert import to_hilbert
from tf_qrnn import QRNN

def calc_custom_loss(pred, gt):
    pass



def ln(tensor, scope=None, epsilon=1e-5):
    """ Layer normalizes a 2D tensor along its second axis """
    assert (len(tensor.get_shape()) == 2)
    m, v = tf.nn.moments(tensor, [1], keep_dims=True)
    if not isinstance(scope, str):
        scope = ''
    with tf.variable_scope(scope + 'layer_norm'):
        scale = tf.get_variable('scale',
                                shape=[tensor.get_shape()[1]],
                                initializer=tf.constant_initializer(1))
        shift = tf.get_variable('shift',
                                shape=[tensor.get_shape()[1]],
                                initializer=tf.constant_initializer(0))
    ln_initial = (tensor - m) / tf.sqrt(v + epsilon)

    return ln_initial * scale + shift


class MultiDimensionalLSTMCell(RNNCell):
    """
    Adapted from TF's BasicLSTMCell to use Layer Normalization.
    Note that state_is_tuple is always True.
    """

    def __init__(self, num_units, forget_bias=0.0, activation=tf.nn.tanh):
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._activation = activation

    @property
    def state_size(self):
        return LSTMStateTuple(self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """Long short-term memory cell (LSTM).
        @param: inputs (batch,n)
        @param state: the states and hidden unit of the two cells
        """
        with tf.variable_scope(scope or type(self).__name__):
            c1, c2, h1, h2 = state

            # change bias argument to False since LN will add bias via shift
            concat = _linear([inputs, h1, h2], 5 * self._num_units, False)

            i, j, f1, f2, o = tf.split(value=concat, num_or_size_splits=5, axis=1)

            # add layer normalization to each gate
            i = ln(i, scope='i/')
            j = ln(j, scope='j/')
            f1 = ln(f1, scope='f1/')
            f2 = ln(f2, scope='f2/')
            o = ln(o, scope='o/')

            new_c = (c1 * tf.nn.sigmoid(f1 + self._forget_bias) +
                     c2 * tf.nn.sigmoid(f2 + self._forget_bias) + tf.nn.sigmoid(i) *
                     self._activation(j))

            # add layer_normalization in calculation of new hidden state
            new_h = self._activation(ln(new_c, scope='new_h/')) * tf.nn.sigmoid(o)
            new_state = LSTMStateTuple(new_c, new_h)

            return new_h, new_state


def multi_dimensional_rnn_while_loop(rnn_size, input_data, sh, dims=None, scope_n="layer1"):
    """Implements naive multi dimension recurrent neural networks

    @param rnn_size: the hidden units
    @param input_data: the data to process of shape [batch,h,w,channels]
    @param sh: [height,width] of the windows
    @param dims: dimensions to reverse the input data,eg.
        dims=[False,True,True,False] => true means reverse dimension
    @param scope_n : the scope

    returns [batch,h/sh[0],w/sh[1],rnn_size] the output of the lstm
    """

    with tf.variable_scope("MultiDimensionalLSTMCell-" + scope_n):
        # Create multidimensional cell with selected size
        cell = MultiDimensionalLSTMCell(rnn_size)

        # Get the shape of the input (batch_size, x, y, channels)
        shape = input_data.get_shape().as_list()
        batch_size = shape[0]
        X_dim = shape[1]
        Y_dim = shape[2]
        channels = shape[3]
        # Window size
        X_win = sh[0]
        Y_win = sh[1]
        # Get the runtime batch size
        batch_size_runtime = tf.shape(input_data)[0]

        # If the input cannot be exactly sampled by the window, we patch it with zeros
        if X_dim % X_win != 0:
            # Get offset size
            offset = tf.zeros([batch_size_runtime, X_win - (X_dim % X_win), Y_dim, channels])
            # Concatenate X dimension
            input_data = tf.concat(axis=1, values=[input_data, offset])
            # Get new shape
            shape = input_data.get_shape().as_list()
            # Update shape value
            X_dim = shape[1]

        # The same but for Y axis
        if Y_dim % Y_win != 0:
            # Get offset size
            offset = tf.zeros([batch_size_runtime, X_dim, Y_win - (Y_dim % Y_win), channels])
            # Concatenate Y dimension
            input_data = tf.concat(axis=2, values=[input_data, offset])
            # Get new shape
            shape = input_data.get_shape().as_list()
            # Update shape value
            Y_dim = shape[2]

        # Get the steps to perform in X and Y axis
        h, w = int(X_dim / X_win), int(Y_dim / Y_win)

        # Get the number of features (total number of imput values per step)
        features = Y_win * X_win * channels

        # Reshape input data to a tensor containing the step indexes and features inputs
        # The batch size is inferred from the tensor size
        x = tf.reshape(input_data, [batch_size_runtime, h, w, features])

        # Reverse the selected dimensions
        if dims is not None:
            assert dims[0] is False and dims[3] is False
            x = tf.reverse(x, dims)

        # Reorder inputs to (h, w, batch_size, features)
        x = tf.transpose(x, [1, 2, 0, 3])
        # Reshape to a one dimensional tensor of (h*w*batch_size , features)
        x = tf.reshape(x, [-1, features])
        # Split tensor into h*w tensors of size (batch_size , features)
        x = tf.split(axis=0, num_or_size_splits=h * w, value=x)

        # Create an input tensor array (literally an array of tensors) to use inside the loop
        inputs_ta = tf.TensorArray(dtype=tf.float32, size=h * w, name='input_ta')
        # Unstack the input X in the tensor array
        inputs_ta = inputs_ta.unstack(x)
        # Create an input tensor array for the states
        states_ta = tf.TensorArray(dtype=tf.float32, size=h * w + 1, name='state_ta', clear_after_read=False)
        # And an other for the output
        outputs_ta = tf.TensorArray(dtype=tf.float32, size=h * w, name='output_ta')

        # initial cell hidden states
        # Write to the last position of the array, the LSTMStateTuple filled with zeros
        states_ta = states_ta.write(h * w, LSTMStateTuple(tf.zeros([batch_size_runtime, rnn_size], tf.float32),
                                                          tf.zeros([batch_size_runtime, rnn_size], tf.float32)))

        # Function to get the sample skipping one row
        def get_up(t_, w_):
            return t_ - tf.constant(w_)

        # Function to get the previous sample
        def get_last(t_, w_):
            return t_ - tf.constant(1)

        # Controls the initial index
        time = tf.constant(0)
        zero = tf.constant(0)

        # Body of the while loop operation that applies the MD LSTM
        def body(time_, outputs_ta_, states_ta_):

            # If the current position is less or equal than the width, we are in the first row
            # and we need to read the zero state we added in row (h*w).
            # If not, get the sample located at a width distance.
            state_up = tf.cond(tf.less_equal(time_, tf.constant(w)),
                               lambda: states_ta_.read(h * w),
                               lambda: states_ta_.read(get_up(time_, w)))

            # If it is the first step we read the zero state if not we read the inmediate last
            state_last = tf.cond(tf.less(zero, tf.mod(time_, tf.constant(w))),
                                 lambda: states_ta_.read(get_last(time_, w)),
                                 lambda: states_ta_.read(h * w))

            # We build the input state in both dimensions
            current_state = state_up[0], state_last[0], state_up[1], state_last[1]
            # Now we calculate the output state and the cell output
            out, state = cell(inputs_ta.read(time_), current_state)
            # We write the output to the output tensor array
            outputs_ta_ = outputs_ta_.write(time_, out)
            # And save the output state to the state tensor array
            states_ta_ = states_ta_.write(time_, state)

            # Return outputs and incremented time step
            return time_ + 1, outputs_ta_, states_ta_

        # Loop output condition. The index, given by the time, should be less than the
        # total number of steps defined within the image
        def condition(time_, outputs_ta_, states_ta_):
            return tf.less(time_, tf.constant(h * w))

        # Run the looped operation
        result, outputs_ta, states_ta = tf.while_loop(condition, body, [time, outputs_ta, states_ta],
                                                      parallel_iterations=1)

        # Extract the output tensors from the processesed tensor array
        outputs = outputs_ta.stack()
        states = states_ta.stack()

        # Reshape outputs to match the shape of the input
        y = tf.reshape(outputs, [h, w, batch_size_runtime, rnn_size])

        # Reorder te dimensions to match the input
        y = tf.transpose(y, [2, 0, 1, 3])
        # Reverse if selected
        if dims is not None:
            y = tf.reverse(y, dims)

        # Return the output and the inner states
        return y, states


def horizontal_standard_lstm(input_data, rnn_size):
    # input is (b, h, w, c)
    b, h, w, c = input_data.get_shape().as_list()
    # transpose = swap h and w.
    new_input_data = tf.reshape(input_data, (b * h, w, c))  # horizontal.
    rnn_out, _ = dynamic_rnn(tf.contrib.rnn.LSTMCell(rnn_size),
                             inputs=new_input_data,
                             dtype=tf.float32)
    rnn_out = tf.reshape(rnn_out, (b, h, w, rnn_size))
    return rnn_out


def snake_standard_lstm(input_data, rnn_size):
    # input is (b, h, w, c)
    b, h, w, c = input_data.get_shape().as_list()
    # transpose = swap h and w.
    new_input_data = tf.reshape(input_data, (b, w * h, c))  # snake.
    rnn_out, _ = dynamic_rnn(tf.contrib.rnn.LSTMCell(rnn_size),
                             inputs=new_input_data,
                             dtype=tf.float32)
    rnn_out = tf.reshape(rnn_out, (b, h, w, rnn_size))
    return rnn_out


def snake_grid_lstm(input_data, rnn_size,scope="snake_rnn1"):
    with tf.variable_scope(scope):
        # input is (b, h, w, c)
        additional_cell_args = {}
        b, h, w, c = input_data.get_shape().as_list()
        cell_fn = grid_rnn.Grid2LSTMCell
        additional_cell_args.update({'use_peepholes': True, 'forget_bias': 1.0,
                                     'state_is_tuple': False, 'output_is_tuple': False})
        cell = cell_fn(rnn_size, **additional_cell_args)
        cell = rnn_cell.MultiRNNCell([cell] * 2)
        # transpose = swap h and w.
        new_input_data = tf.reshape(input_data, (b, w * h, c))  # snake.
        rnn_out, _ = dynamic_rnn(cell,
                                 inputs=new_input_data,
                                 dtype=tf.float32)
        rnn_out = tf.reshape(rnn_out, (b, h, w, rnn_size))
        return rnn_out



def hilbert_grid_lstm(input_data, rnn_size,scope="snake_rnn1"):
    with tf.variable_scope(scope):
        # input is (b, h, w, c)
        additional_cell_args = {}
        b, h, w, c = input_data.get_shape().as_list()
        cell_fn = grid_rnn.Grid2LSTMCell
        additional_cell_args.update({'use_peepholes': True, 'forget_bias': 1.0,
                                     'state_is_tuple': False, 'output_is_tuple': False})
        cell = cell_fn(rnn_size, **additional_cell_args)
        # transpose = swap h and w.
        new_input_data = to_hilbert(input_data, b, h, w, c)#tf.reshape(input_data, (b, w * h, c))  # snake.
        print("new_input_data shape: "+str(new_input_data.shape))
        rnn_out, _ = dynamic_rnn(cell,
                                 inputs=new_input_data,
                                 dtype=tf.float32)
        rnn_out = tf.reshape(rnn_out, (b, h, w, rnn_size))
        return rnn_out

def md_hilbert_grid_lstm(input_data, rnn_size):
    """
    Fragen:
    shape input_data A:input (batch_size, x, y, channels)
    tensorflow rotation input daten
    """
    #1. Direction
    #print("1.")
    #print(input_data.shape)
    out1 = hilbert_grid_lstm(input_data, rnn_size, scope="hilbert_rnn1")

    #2. Direction
    #print("2.")

    #print(new_input_data.shape)
    """
    input_data_2 = slim.fully_connected(inputs=new_input_data,
                                     num_outputs=1,
                                     activation_fn=tf.nn.sigmoid)
    """
    #for i in range(new_input_data)
    input_data = tf.transpose(input_data,perm=[0, 2, 1, 3])
    input_data = tf.reverse(input_data,[2])#np.flipud(input_data_2)
    #print(input_data_2.shape)
    out2 = hilbert_grid_lstm(input_data, rnn_size, scope="hilbert_rnn2")

    #3. Direction
    #print("3.")
    #print(new_input_data.shape)
    """
    input_data_3 = slim.fully_connected(inputs=new_input_data,
                                     num_outputs=1,
                                     activation_fn=tf.nn.sigmoid)
    """
    input_data = tf.transpose(input_data,perm=[0, 2, 1, 3])
    input_data = tf.reverse(input_data,[2])#np.flipud(input_data_3)
    #print(input_data_3.shape)
    out3 = hilbert_grid_lstm(input_data, rnn_size, scope="hilbert_rnn3")

    #4. Direction
    #print("4.")
    #print(new_input_data.shape)
    """
    input_data_4 = slim.fully_connected(inputs=new_input_data,
                                     num_outputs=1,
                                     activation_fn=tf.nn.sigmoid)
    """
    input_data = tf.transpose(input_data,perm=[0, 2, 1, 3])
    input_data = tf.reverse(input_data,[2])#np.flipud(input_data_4)
    #print(input_data_3.shape)
    out4 = hilbert_grid_lstm(input_data, rnn_size, scope="hilbert_rnn4")
    #prepare output

    #print(output.shape)
    #output = tf.transpose(output,perm=[0, 2, 1, 3])
    #print(output.shape)
    #output = tf.reverse(output,[2])#np.flipud(input_data_4)
    #print(output.shape)
    output = out1
    output = tf.math.maximum(output,out2)
    output = tf.math.maximum(output,out3)
    rnn_out = tf.math.maximum(output,out4)

    return rnn_out

def md_snake_grid_lstm(input_data, rnn_size):
    """
    Fragen:
    shape input_data A:input (batch_size, x, y, channels)
    tensorflow rotation input daten
    """
    #1. Direction
    #print("1.")
    #print(input_data.shape)
    out1 = snake_grid_lstm(input_data, rnn_size, scope="snake_rnn1")

    #2. Direction
    #print("2.")

    #print(new_input_data.shape)
    """
    input_data_2 = slim.fully_connected(inputs=new_input_data,
                                     num_outputs=1,
                                     activation_fn=tf.nn.sigmoid)
    """
    #for i in range(new_input_data)
    input_data = tf.transpose(input_data,perm=[0, 2, 1, 3])
    input_data = tf.reverse(input_data,[2])#np.flipud(input_data_2)
    #print(input_data_2.shape)
    out2 = snake_grid_lstm(input_data, rnn_size, scope="snake_rnn2")

    #3. Direction
    #print("3.")
    #print(new_input_data.shape)
    """
    input_data_3 = slim.fully_connected(inputs=new_input_data,
                                     num_outputs=1,
                                     activation_fn=tf.nn.sigmoid)
    """
    input_data = tf.transpose(input_data,perm=[0, 2, 1, 3])
    input_data = tf.reverse(input_data,[2])#np.flipud(input_data_3)
    #print(input_data_3.shape)
    out3 = snake_grid_lstm(input_data, rnn_size, scope="snake_rnn3")

    #4. Direction
    #print("4.")
    #print(new_input_data.shape)
    """
    input_data_4 = slim.fully_connected(inputs=new_input_data,
                                     num_outputs=1,
                                     activation_fn=tf.nn.sigmoid)
    """
    input_data = tf.transpose(input_data,perm=[0, 2, 1, 3])
    input_data = tf.reverse(input_data,[2])#np.flipud(input_data_4)
    #print(input_data_3.shape)
    out4 = snake_grid_lstm(input_data, rnn_size, scope="snake_rnn4")
    #prepare output

    #print(output.shape)
    #output = tf.transpose(output,perm=[0, 2, 1, 3])
    #print(output.shape)
    #output = tf.reverse(output,[2])#np.flipud(input_data_4)
    #print(output.shape)
    output = out1
    output = tf.math.maximum(output,out2)
    output = tf.math.maximum(output,out3)
    rnn_out = tf.math.maximum(output,out4)

    return rnn_out

def md_qrnn_combi(input_data, rnn_size, size, in_size, batch_size):
    tmp_result, _ = multi_dimensional_rnn_while_loop(rnn_size=rnn_size, input_data=input_data, sh=[1, 1])

    model_out = slim.fully_connected(inputs=tmp_result,
                                 num_outputs=1,
                                 activation_fn=tf.nn.sigmoid)

    conv_size = rnn_size
    print("QRNN: "+str(batch_size)+" "+str(in_size)+" "+str(size*size)+" "+str(conv_size))
    qrnn = QRNN(b_size=batch_size,in_size=in_size, size=size*size, conv_size=conv_size)
    #qrnn = QRNN(in_size=word_size, size=size, conv_size=3)
    #x_reshaped = tf.squeeze(tf.reshape(x,[batch_size, in_w*in_h, channels]))
    print(model_out)
    x_reshaped = tf.reshape(model_out,[batch_size, size*size, in_size])
    #x_reshaped = tf.squeeze(x_reshaped)
    print(x_reshaped)
    rnn_out = qrnn.forward(x_reshaped)
    rnn_out = tf.reshape(rnn_out,[batch_size, size, size, in_size])
    print("rnn_out:")
    print(rnn_out)

    return rnn_out

def md_qrnn_combi2(input_data, rnn_size, size, in_size, batch_size):
    conv_size = rnn_size
    print("QRNN: "+str(batch_size)+" "+str(in_size)+" "+str(size*size)+" "+str(conv_size))
    qrnn = QRNN(b_size=batch_size,in_size=in_size, size=size*size, conv_size=conv_size)
    #qrnn = QRNN(in_size=word_size, size=size, conv_size=3)
    #x_reshaped = tf.squeeze(tf.reshape(x,[batch_size, in_w*in_h, channels]))

    x_reshaped = tf.reshape(input_data,[batch_size, size*size, in_size])
    #x_reshaped = tf.squeeze(x_reshaped)
    print(x_reshaped)
    rnn_out = qrnn.forward(x_reshaped)
    qrnn_result = tf.reshape(rnn_out,[batch_size, size, size, in_size])
    print("rnn_out:")
    print(rnn_out)

    md_result, _ = multi_dimensional_rnn_while_loop(rnn_size=rnn_size, input_data=input_data, sh=[1, 1])

    qrnn_slim = slim.fully_connected(inputs=qrnn_result,
                                 num_outputs=1,
                                 activation_fn=tf.nn.sigmoid)

    md_slim = slim.fully_connected(inputs=md_result,
                                 num_outputs=1,
                                 activation_fn=tf.nn.sigmoid)

    model_out = slim.fully_connected(inputs=tf.stack([qrnn_slim,md_slim],axis=4),
                                 num_outputs=1,
                                 activation_fn=tf.nn.sigmoid)
    print(model_out)
    return model_out


def resnet(input_data, resnet_size, input_size, training):
    net = DeepLabResNetModel({'data': input_data}, is_training=True, num_classes=2)
    # For a small batch size, it is better to keep
    # the statistics of the BN layers (running means and variances)
    # frozen, and to not update the values provided by the pre-trained model.
    # If is_training=True, the statistics will be updated during the training.
    # Note that is_training=False still updates BN parameters gamma (scale) and beta (offset)
    # if they are presented in var_list of the optimiser definition.

    # Predictions.
    raw_output = net.layers['fc1_voc12']

    #raw_prediction = tf.reshape(raw_output, [-1, 2])
    #label_proc = prepare_label(label_batch, tf.stack(raw_output.get_shape()[1:3]), num_classes=2, one_hot=False) # [batch_size, h, w]
    #raw_gt = tf.reshape(label_proc, [-1,])
    #indices = tf.squeeze(tf.where(tf.less_equal(raw_prediction, 2-1)), 1)
    #gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
    #resnet_out = tf.gather(raw_prediction, indices)
    print("raw_output: "+str(raw_output))

    return raw_output
