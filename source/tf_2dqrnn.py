import tensorflow as tf


class DQRNN():
    def __init__(self,b_size, in_size, x_size, y_size, conv_size=2):
        self.kernel = None
        self.batch_size = b_size #-1
        self.conv_size = conv_size
        self.c_matrix = tf.zeros((y_size,x_size))#np.ones((x_size, y_size))
        self.c = 1
        self.c_1 = 1 #None
        self.c_2 = 1 #None
        self.h = None
        self._x = None
        if conv_size == 1:
            self.kernel = DQRNNLinear(in_size, x_size, y_size)
        elif conv_size == 2:
            self.kernel = DQRNNWithPrevious(in_size, x_size, y_size)
        else:
            print("Sizes:")
            print(in_size)
            print(x_size)
            print(y_size)
            print(conv_size)
            self.kernel = DQRNNConvolution(in_size, x_size, y_size, conv_size)
    """
    def _step(self, f, z, o, i_y, i_x):
        with tf.variable_scope("fo-Pool"):
            # f,z,o is batch_size x size
            f = tf.sigmoid(f)
            z = tf.tanh(z)
            o = tf.sigmoid(o)
            #print("#####Pooling shapes#####\n")
            #print("f: "+str(f))
            #print("z: "+str(z))
            #print("o: "+str(o))

            #self.c = tf.multiply(f, self.c) + tf.multiply(1 - f, z)
            #self.c_1 = tf.multiply(f, self.c) + tf.multiply(1 - f, z)
            #self.c_1 = tf.multiply(f, self.c) + tf.multiply(1 - f, z)
            if i_y == 0:
                self.c_matrix[i_y,i_x] = tf.multiply(f, 1) + tf.multiply(f, self.c_matrix[i_y,i_x-1]) + tf.multiply(1 - f, z)                      #tf.reduce_mean(self.c_1, self.c_2)
            elif i_x == 0:
                self.c_matrix[i_y,i_x] = tf.multiply(f, self.c_matrix[i_y-1,i_x]) + tf.multiply(f, 1) + tf.multiply(1 - f, z)
            elif i_y == 0 and i_x == 0:
                self.c_matrix[i_y,i_x] = tf.multiply(f, 1) + tf.multiply(f, 1) + tf.multiply(1 - f, z)
            else:
                self.c_matrix[i_y,i_x] = tf.multiply(f, self.c_matrix[i_y-1,i_x]) + tf.multiply(f,  self.c_matrix[i_y,i_x-1]) + tf.multiply(1 - f, z)

            self.h = tf.multiply(o, self.c_matrix[i_y,i_x])     #self.h = tf.multiply(o, self.c)  # h is size vector

        return self.h
        """
    def _step_y(self, f, z, o, i_y, i_x):
        with tf.variable_scope("fo-Pool"):
            self.c_1 = tf.multiply(f,  self.c_1) + tf.multiply(1 - f, z)

            #self.h = tf.multiply(o, self.c_matrix[i_y,i_x])     #self.h = tf.multiply(o, self.c)  # h is size vector

        return self.c_1

    def _step_x(self, f, z, o, i_y, i_x):
        with tf.variable_scope("fo-Pool"):
            self.c_2 = tf.multiply(f,  self.c_2) + tf.multiply(1 - f, z)

            #self.h = tf.multiply(o, self.c_matrix[i_y,i_x])     #self.h = tf.multiply(o, self.c)  # h is size vector

        return self.c_2

    def forward(self, x):
        length = lambda mx: int(mx.get_shape()[0])

        with tf.variable_scope("QRNN/Forward"):
            """
            if self.c is None:
                # init context cell
                self.c = tf.zeros([length(x), self.kernel.size], dtype=tf.float32)
            """

            if self.conv_size <= 2:
                # x is batch_size x sentence_length x word_length
                # -> now, transpose it to sentence_length x batch_size x word_length
                _x = tf.transpose(x, [1, 0, 2])

                for i in range(length(_x)):
                    t = _x[i] # t is batch_size x word_length matrix
                    f, z, o = self.kernel.forward(t)
                    self._step(f, z, o)
            else:
                c_f, c_z, c_o = self.kernel.conv(x)
                print("#####C SHAPES#####\n")
                print("c_f: "+str(c_f))
                print("c_z: "+str(c_z))
                print("c_o: "+str(c_o))
                for i_y in range(length(c_f)):                                  #CHANGED
                    #print(length(c_f))
                    for i_x in range(length(c_f[i_y])):
                        #print(length(c_f[i_y]))
                        f, z, o = c_f[i_y][i_x], c_z[i_y][i_x], c_o[i_y][i_x]
                        self._step_y(f, z, o, i_y, i_x)
                        self._step_x(f, z, o, i_y, i_x)
                        self.h = tf.multiply(o, tf.multiply(tf.multiply(self.c_1,self.c_2),0.5))

        return self.h


class DQRNNLinear():

    def __init__(self, in_size, x_size, y_size):
        self.in_size = in_size
        self.size = size
        self._weight_size = self.size * 3  # z, f, o
        with tf.variable_scope("QRNN/Variable/Linear"):
            initializer = tf.random_normal_initializer()
            self.W = tf.get_variable("W", [self.in_size, self._weight_size], initializer=initializer)
            self.b = tf.get_variable("b", [self._weight_size], initializer=initializer)

    def forward(self, t):
        # x is batch_size x word_length matrix
        _weighted = tf.matmul(t, self.W)
        _weighted = tf.add(_weighted, self.b)

        # now, _weighted is batch_size x weight_size
        f, z, o = tf.split(_weighted, num_or_size_splits=3, axis=1)  # split to f, z, o. each matrix is batch_size x size
        return f, z, o


class DQRNNWithPrevious():

    def __init__(self, in_size, x_size, y_size):
        self.in_size = in_size
        self.size = size
        self._weight_size = self.size * 3  # z, f, o
        self._previous = None
        with tf.variable_scope("QRNN/Variable/WithPrevious"):
            initializer = tf.random_normal_initializer()
            self.W = tf.get_variable("W", [self.in_size, self._weight_size], initializer=initializer)
            self.V = tf.get_variable("V", [self.in_size, self._weight_size], initializer=initializer)
            self.b = tf.get_variable("b", [self._weight_size], initializer=initializer)

    def forward(self, t):
        if self._previous is None:
            self._previous = tf.get_variable("previous", [t.get_shape()[0], self.in_size], initializer=tf.random_normal_initializer())

        _current = tf.matmul(t, self.W)
        _previous = tf.matmul(self._previous, self.V)
        _previous = tf.add(_previous, self.b)
        _weighted = tf.add(_current, _previous)

        f, z, o = tf.split(_weighted, num_or_size_splits=3, axis=1)  # split to f, z, o. each matrix is batch_size x size
        self._previous = t
        return f, z, o


class DQRNNConvolution():

    def __init__(self, in_size, size_x, size_y, conv_size):
        self.in_size = in_size
        self.size_x = size_x
        self.size_y = size_y
        self.conv_size = conv_size
        self._weight_size = self.size_x * self.size_y * 3  # z, f, o            CHANGED

        with tf.variable_scope("DQRNN/Variable/Convolution"):
            initializer = tf.random_normal_initializer()
            self.conv_filter = tf.get_variable("conv_filter", [conv_size, in_size, in_size, self._weight_size], initializer=initializer)

    def conv(self, x):
        """
        AUSGABE AUS 1D:
        Tensor("Reshape:0", shape=(16, 625, 1), dtype=float32)
        #####
        Tensor("Reshape:0", shape=(16, 625, 1), dtype=float32)
        <tf.Variable 'QRNN/Variable/Convolution/conv_filter:0' shape=(2500, 1, 1875) dtype=float32_ref>
        #####

        _weighted: Tensor("QRNN/Forward/conv1d/Squeeze:0", shape=(16, 625, 1875), dtype=float32)
        transpose: Tensor("QRNN/Forward/transpose:0", shape=(625, 16, 1875), dtype=float32)
        rnn_out: Tensor("Reshape_1:0", shape=(16, 25, 25, 1), dtype=float32)
        pool_out: Tensor("Reshape_1:0", shape=(16, 25, 25, 1), dtype=float32)
        reshape_out: Tensor("Reshape_2:0", shape=(16, 25, 25, 1), dtype=float32)
        """
        # !! x is batch_size x y_size x x_size x word_length(=channel) !!
        print("#####")
        print(x)
        print(self.conv_filter)
        print("#####\n")
        _weighted = tf.nn.conv2d(x, self.conv_filter, strides=[1,1,1,1], padding="SAME")
        print("_weighted: "+str(_weighted))

        # _weighted is batch_size x conved_size x output_channel
        _w = tf.transpose(_weighted, [1,2,0,3])  # conved_size x  batch_size x output_channel
        print("transpose: "+str(_w))
        _ws = tf.split(_w, num_or_size_splits=3, axis=3) # make 3(f, z, o) conved_size x  batch_size x size
        return _ws
