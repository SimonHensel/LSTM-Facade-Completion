#import os
#os.system("pip install --upgrade pip")
#os.system("apt-get update")
#os.system("apt-get install -y libsm6 libxext6 libxrender-dev")
#os.system("pip install numpy opencv-python hilbertcurve")

from time import time
import os
import math

import argparse
import logging
import numpy as np
import tensorflow.contrib.slim as slim
from enum import Enum
from tf_qrnn import QRNN
from pathlib import Path

from data_cmp_generate import interpolate, next_batch, write_mat, write_coordinates
#from generate_low_res_facade import next_batch, write_mat, get_relevant_prediction_index

from md_lstm import *
#from own_loss_test import shape_loss_batch, distbce_loss_batch
from mdmd_lstm import multi_directional_md_rnn_while_loop

logger = logging.getLogger(__name__)

def prep_batch_resnet(input_data): # shape (16,512,512,1)
    out = []
    for i in range(input_data.shape[0]):
        out.append(interpolate(input_data[i],512))

    return np.array(out)


def get_script_arguments():
    parser = argparse.ArgumentParser(description='MD LSTM trainer.')
    parser.add_argument('--model_type', required=True, type=ModelType.from_string,
                        choices=list(ModelType), help='Model type.')
    parser.add_argument('--enable_plotting', action='store_true')

    parser.add_argument('--checkpoint_path', help='checkpoint_path')

    parser.add_argument('--loss_type',type=LossType.from_string,
                        choices=list(LossType), help='defines loss function')

    args = get_arguments(parser)
    logger.info('Script inputs: {}.'.format(args))
    return args


class FileLogger(object):
    def __init__(self, full_filename, headers):
        self._headers = headers
        self._out_fp = open(full_filename, 'w')
        self._write(headers)

    def write(self, line):
        assert len(line) == len(self._headers)
        self._write(line)

    def close(self):
        self._out_fp.close()

    def _write(self, arr):
        arr = [str(e) for e in arr]
        self._out_fp.write(' '.join(arr) + '\n')
        self._out_fp.flush()


class ModelType(Enum):
    MDMD_LSTM = 'MDMD_LSTM'
    MD_LSTM = 'MD_LSTM'
    MD_LSTM_DISTANCE = 'MD_LSTM_DISTANCE'
    MDMD_LSTM_DISTANCE = 'MDMD_LSTM_DISTANCE'
    HORIZONTAL_SD_LSTM = 'HORIZONTAL_SD_LSTM'
    SNAKE_SD_LSTM = 'SNAKE_SD_LSTM'
    SNAKE_GRID_LSTM = 'SNAKE_GRID_LSTM'
    HILBERT_GRID_LSTM = 'HILBERT_GRID_LSTM'
    MD_HILBERT_GRID_LSTM = 'MD_HILBERT_GRID_LSTM'
    MD_SNAKE_GRID_LSTM = 'MD_SNAKE_GRID_LSTM'
    MD_SNAKE_GRID_LSTM_C = 'MD_SNAKE_GRID_LSTM_C'
    RESNET = 'RESNET'
    QRNN = 'QRNN'
    MD_QRNN_COMBI = 'MD_QRNN_COMBI'
    MD_QRNN_COMBI2 = 'MD_QRNN_COMBI2'
    DQRNN = 'DQRNN'

    def __str__(self):
        return self.value

    @staticmethod
    def from_string(s):
        try:
            return ModelType[s]
        except KeyError:
            raise ValueError()

class LossType(Enum):
    DEFAULT = "DEFAULT"
    SHAPE_COMBI = "SHAPE_COMBI"
    SHAPE_ONLY = "SHAPE_ONLY"

    def __str__(self):
        return self.value

    @staticmethod
    def from_string(s):
        try:
            return LossType[s]
        except KeyError:
            raise ValueError()

def get_arguments(parser: argparse.ArgumentParser):
    args = None
    try:
        args = parser.parse_args()
    except Exception:
        parser.print_help()
        exit(1)
    return args




def run(model_type='md_lstm',enable_plotting=True, checkpoint_path="checkpoint/model.ckpt", loss_type='DEFAULT'):
    #run(args.model_type, args.enable_plotting, args.checkpoint_path, args.loss_type)
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
    learning_rate = 0.01
    batch_size = 16
    in_h = 25#32#128
    in_w = 25#32#128
    out_h = 25#32#128
    out_w = 25#32#128
    channels = 1
    #hidden_size = 2500#4096#1024#2048#4096#256#128#64#256 #2500 for size 25

    x = tf.placeholder(tf.float32, [batch_size, in_h, in_w, channels])


    if model_type == ModelType.MD_LSTM:
        hidden_size = 2500#4096#1024#2048#4096#256#128#64#256 #2500 for size 25 #5000 for P6000
        print(model_type)
        y = tf.placeholder(tf.float32, [batch_size, out_h, out_w, channels])
        logger.info('Using Multi Dimensional LSTM.')
        rnn_out, _ = multi_dimensional_rnn_while_loop(rnn_size=hidden_size, input_data=x, sh=[1, 1])
    elif model_type == ModelType.MDMD_LSTM:
        hidden_size = 1250 #2500 for P6000
        print(model_type)
        y = tf.placeholder(tf.float32, [batch_size, out_h, out_w, channels])
        logger.info('Using Multi Dimensional LSTM.')
        rnn_out, _ = multi_directional_md_rnn_while_loop(rnn_size=hidden_size, input_data=x, sh=[1, 1])
    elif model_type == ModelType.HORIZONTAL_SD_LSTM:
        hidden_size = 256
        print(model_type)
        y = tf.placeholder(tf.float32, [batch_size, out_h, out_w, channels])
        logger.info('Using Standard LSTM.')
        rnn_out = horizontal_standard_lstm(input_data=x, rnn_size=hidden_size)
    elif model_type == ModelType.SNAKE_SD_LSTM:
        hidden_size = 256
        print(model_type)
        y = tf.placeholder(tf.float32, [batch_size, out_h, out_w, channels])
        rnn_out = snake_standard_lstm(input_data=x, rnn_size=hidden_size)
    elif model_type == ModelType.SNAKE_GRID_LSTM:
        hidden_size = 2500
        print(model_type)
        y = tf.placeholder(tf.float32, [batch_size, out_h, out_w, channels])
        rnn_out = snake_grid_lstm(input_data=x, rnn_size=hidden_size)
        print("\n")
        print(rnn_out)
        print("\n")
    elif model_type == ModelType.HILBERT_GRID_LSTM:
        hidden_size = 256
        print(model_type)
        y = tf.placeholder(tf.float32, [batch_size, out_h, out_w, channels])
        rnn_out = hilbert_grid_lstm(input_data=x, rnn_size=hidden_size)
    elif model_type == ModelType.MD_SNAKE_GRID_LSTM:
        hidden_size = 256
        print(model_type)
        y = tf.placeholder(tf.float32, [batch_size, out_h, out_w, channels])
        rnn_out = md_snake_grid_lstm(input_data=x, rnn_size=hidden_size)
    elif model_type == ModelType.MD_HILBERT_GRID_LSTM:
        hidden_size = 256
        print(model_type)
        y = tf.placeholder(tf.float32, [batch_size, out_h, out_w, channels])
        rnn_out = md_hilbert_grid_lstm(input_data=x, rnn_size=hidden_size)
    elif model_type == ModelType.MD_SNAKE_GRID_LSTM_C:
        print(model_type)
        learning_rate = 0.01
        batch_size = 1
        in_h = 64#32#128
        in_w = 64#32#128
        #out_h = 16#32#128
        out_len = 64#32#128
        max_pool_h = int(math.sqrt(in_h))
        max_pool_w = int(math.sqrt(in_w))
        channels = 1
        hidden_size = 256#128#64#256
        y = tf.placeholder(tf.float32, [batch_size, out_len, channels])
        rnn_out = md_snake_grid_lstm(input_data=x, rnn_size=hidden_size)

    elif model_type == ModelType.RESNET:
        print(model_type)
        learning_rate = 0.01
        batch_size = 6
        #in_h = 64#32#128
        input_size = 8*64#32#128
        #out_h = 16#32#128
        out_len = 64#32#128
        max_pool_h = int(math.sqrt(in_h))
        max_pool_w = int(math.sqrt(in_w))
        channels = 1
        hidden_size = 256#128#64#256
        x = tf.placeholder(tf.float32, [batch_size, input_size, input_size, channels])
        y = tf.placeholder(tf.float32, [batch_size, out_len, out_len, channels])
        rnn_out = resnet(
            input_data=x,
            resnet_size=hidden_size,
            input_size=input_size,
            training=True
        )
    elif model_type == ModelType.HILBERT_GRID_LSTM:
        hidden_size = 256
        print(model_type)
        y = tf.placeholder(tf.float32, [batch_size, out_h, out_w, channels])
        rnn_out = hilbert_grid_lstm(input_data=x, rnn_size=hidden_size)
    elif model_type == ModelType.MD_LSTM_DISTANCE:
        hidden_size = 256
        print(model_type)
        y = tf.placeholder(tf.float32, [batch_size, out_h, out_w, channels])
        y_dist1 = tf.placeholder(tf.float32, [batch_size, out_h, out_w, channels])
        y_dist2 = tf.placeholder(tf.float32, [batch_size, out_h, out_w, channels])
        rnn_out, _ = multi_dimensional_rnn_while_loop(input_data=x, rnn_size=hidden_size,sh=[1, 1])
    elif model_type == ModelType.MDMD_LSTM_DISTANCE:
        hidden_size = 256
        print(model_type)
        y = tf.placeholder(tf.float32, [batch_size, out_h, out_w, channels])
        y_dist1 = tf.placeholder(tf.float32, [batch_size, out_h, out_w, channels])
        y_dist2 = tf.placeholder(tf.float32, [batch_size, out_h, out_w, channels])
        rnn_out, _ = multi_directional_md_rnn_while_loop(input_data=x, rnn_size=hidden_size,sh=[1, 1])
    elif model_type == ModelType.QRNN:
        hidden_size = 2500
        y = tf.placeholder(tf.float32, [batch_size, out_h, out_w, channels])
        logger.info('Using QRNN.')
        size = in_h
        in_size = channels
        conv_size = hidden_size
        qrnn = QRNN(b_size=batch_size,in_size=in_size, size=size*size, conv_size=conv_size)
        #qrnn = QRNN(in_size=word_size, size=size, conv_size=3)
        #x_reshaped = tf.squeeze(tf.reshape(x,[batch_size, in_w*in_h, channels]))
        x_reshaped = tf.reshape(x,[batch_size, in_w*in_h, channels])
        #x_reshaped = tf.squeeze(x_reshaped)
        print(x_reshaped)
        rnn_out = qrnn.forward(x_reshaped)
        rnn_out = tf.reshape(rnn_out,[batch_size, in_w, in_h, channels])
        print("rnn_out:")
        print(rnn_out)
    elif model_type == ModelType.MD_QRNN_COMBI:
        hidden_size = 1250
        print(model_type)
        size = in_h
        in_size = channels
        y = tf.placeholder(tf.float32, [batch_size, out_h, out_w, channels])
        logger.info('Using Multi Dimensional LSTM.')
        rnn_out = md_qrnn_combi(input_data=x, rnn_size=hidden_size, size=size, in_size=in_size, batch_size=batch_size)
    elif model_type == ModelType.MD_QRNN_COMBI2:
        hidden_size = 1250
        print(model_type)
        size = in_h
        in_size = channels
        y = tf.placeholder(tf.float32, [batch_size, out_h, out_w, channels])
        logger.info('Using Multi Dimensional LSTM.')
        rnn_out = md_qrnn_combi2(input_data=x, rnn_size=hidden_size, size=size, in_size=in_size, batch_size=batch_size)
    elif model_type == ModelType.DQRNN:
        hidden_size = 1250
        y = tf.placeholder(tf.float32, [batch_size, out_h, out_w, channels])
        logger.info('Using DQRNN.')
        #size = in_h
        in_size = in_h
        conv_size = hidden_size
        qrnn = DQRNN(b_size=batch_size,in_size=1, x_size=in_size, y_size=in_size, conv_size=conv_size)
        #qrnn = QRNN(in_size=word_size, size=size, conv_size=3)
        #x_reshaped = tf.squeeze(tf.reshape(x,[batch_size, in_w*in_h, channels]))
        x_reshaped = x#tf.reshape(x,[batch_size, in_w*in_h, channels])
        #x_reshaped = tf.squeeze(x_reshaped)
        print(x_reshaped)
        rnn_out = qrnn.forward(x_reshaped)
        print("qrnn.forward: "+str(rnn_out)+"\n") #shape=(25, 16, 625)
        rnn_out = tf.reshape(rnn_out,[batch_size, in_w, in_h, channels])
        print("rnn_out:")
        print(rnn_out)
    else:
        raise Exception('Unknown model type: {}.'.format(model_type))

    #print(y)
    logits = rnn_out
    if not (model_type == ModelType.QRNN or model_type == ModelType.MD_QRNN_COMBI or model_type == ModelType.DQRNN):
        model_out = slim.fully_connected(inputs=rnn_out,
                                     num_outputs=1,
                                     activation_fn=tf.nn.sigmoid)
    else:
        model_out = rnn_out


    if model_type == ModelType.MD_SNAKE_GRID_LSTM_C:
        pool_out = tf.layers.max_pooling2d(model_out,(max_pool_h,max_pool_w),8)
    else:
        pool_out = model_out

    print("pool_out: "+str(pool_out))
    reshape_out = tf.reshape(pool_out, y.shape)
    print("reshape_out: "+str(reshape_out))

    print("#########LOSS#CALC############")
    #print(y)
    #print(reshape_out)
    print(loss_type)

    if loss_type == LossType.DEFAULT and (model_type != ModelType.MD_LSTM_DISTANCE or model_type == ModelType.MDMD_LSTM_DISTANCE):
        print("USING DEFAULT LOSS...")
        sh_loss = tf.constant(0.0)
        all_loss = tf.reduce_mean(tf.square(y - reshape_out))
        loss = all_loss
    elif loss_type == LossType.SHAPE_COMBI:
        print("USING SHAPE_COMBI LOSS...")
        loss = tf.reduce_mean(tf.square(y - reshape_out))
        sh_loss = shape_loss_batch(reshape_out,y, debug=False)
        #print(loss)
        #print(sh_loss)
        all_loss = tf.add(loss,sh_loss)
    elif loss_type == LossType.SHAPE_ONLY:
        print("USING SHAPE_ONLY LOSS...")
        loss = tf.constant(0.0)
        all_loss = shape_loss_batch(reshape_out,y, debug=False)
        sh_loss = all_loss
    elif loss_type == LossType.DEFAULT and (model_type == ModelType.MD_LSTM_DISTANCE or model_type == ModelType.MDMD_LSTM_DISTANCE):
        print("USING BCE DISTANCE LOSS...")
        sh_loss = tf.constant(0.0)

        all_loss = distbce_loss_batch(reshape_out,y, y_dist1, y_dist2, logits,debug=False)#tf.losses.log_loss(y,reshape_out)#distbce_loss_batch(reshape_out,y, y_dist1, y_dist2, logits,debug=False)
        #all_loss = tf.reduce_mean(all_loss)
        loss = all_loss

    print("Hier Fehler")
    """
    Hier Fehler
    Tensor("truediv_16:0", shape=(2, 64, 64), dtype=float32)
    """
    print(all_loss)
    grad_update = tf.train.AdamOptimizer(learning_rate).minimize(all_loss)
    gpu_options = tf.GPUOptions(allow_growth = True)
    # Add ops to save and restore all the variables.
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False,gpu_options=gpu_options))
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    # Restore variables from disk.
    if os.path.isfile(checkpoint_path+".index"):
        saver.restore(sess, checkpoint_path)
        print("Model restored.")
    else:
        print("\n WARNING: NO SAVE FOUND IN "+str(checkpoint_path)+"\n")


    fp = FileLogger('out_{}.tsv'.format(model_type), ['steps_{}'.format(model_type),
                                                      'overall_loss_{}'.format(model_type),
                                                      'time_{}'.format(model_type),
                                                      'relevant_loss_{}'.format(model_type)])

    steps = 20000+1
    for i in range(steps):
        batch = next_batch(batch_size)

        grad_step_start_time = time()
        batch_x = np.expand_dims(batch[0], axis=3)
        if model_type == ModelType.RESNET:
            batch_x = prep_batch_resnet(batch_x)
            batch_x = np.expand_dims(batch_x, axis=3)
        batch_y = np.expand_dims(batch[1], axis=3)
        #print(batch_x.shape)
        #print(batch_y.shape)

        model_preds, tot_loss_value, r_sh_loss, r_g_loss, _ = sess.run([model_out, all_loss, sh_loss, loss, grad_update], feed_dict={x: batch_x, y: batch_y})


        relevant_loss = 0.0

        values = [str(i).zfill(4) , time() - grad_step_start_time, tot_loss_value, r_sh_loss, r_g_loss]
        format_str = '{0} | time {1:.3f} \nall loss = {2:.3f} | g loss = {3:.3f} |shape loss = {4:.3f} |\n'
        if math.isnan(tot_loss_value):
            print("ERROR: NAN in Loss!")
            exit()

        logger.info(format_str.format(*values))
        #fp.write(values)

        display_matplotlib_every = 50
        if i % 100 == 0:
            save_path = checkpoint_path#"checkpoint/model.ckpt"
            saver.save(sess, save_path)
            print("Model saved in path: %s" % save_path)
        if i % display_matplotlib_every == 0 or (i % display_matplotlib_every == 0): #MORE OUTPUTS AFTER A AMOUNT OF ITERATIONS
            x_name = "x_"+'{:06d}'.format(i)+".png"
            y_name = "y_"+'{:06d}'.format(i)+".png"
            z_name = "z_"+'{:06d}'.format(i)+".png"
            print("Shape batch: "+str(batch_x[0].shape))
            #print(batch_x[0])
            write_mat(batch_x[0].squeeze(), x_name)


            if model_type == ModelType.MD_SNAKE_GRID_LSTM_C:
                write_coordinates(batch_y[0].squeeze(), batch_x[0].squeeze().shape , y_name)
                write_out = sess.run(reshape_out, feed_dict={x: batch_x})[0].squeeze()
                print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                print(write_out.shape)
                write_coordinates(write_out, batch_x[0].squeeze().shape, z_name)
            else:
                write_mat(sess.run(model_out, feed_dict={x: batch_x})[0].squeeze(), z_name)
                write_mat(batch_y[0].squeeze(), y_name)



def main():
    args = get_script_arguments()
    logging.basicConfig(format='%(asctime)12s - %(levelname)s - %(message)s', level=logging.INFO)
    run(args.model_type, args.enable_plotting, args.checkpoint_path, args.loss_type)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    main()
