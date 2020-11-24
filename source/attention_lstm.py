import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell, LSTMStateTuple
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import _linear
from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.contrib import grid_rnn
from resnet import DeepLabResNetModel
from to_hilbert import to_hilbert

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

def attention_standard_lstm(input_data, rnn_size):
    # input is (b, h, w, c)
    b, h, w, c = input_data.get_shape().as_list()
    # transpose = swap h and w.
    new_input_data = tf.reshape(input_data, (b * h, w, c))  # TODO Replace this with attention
    """
    Was brauchen wir:
        - vertikale Attention
        - horizontalle Attention
    """
    rnn_out, _ = dynamic_rnn(tf.contrib.rnn.LSTMCell(rnn_size),
                             inputs=new_input_data,
                             dtype=tf.float32)
    rnn_out = tf.reshape(rnn_out, (b, h, w, rnn_size))
    return rnn_out
