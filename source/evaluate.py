from time import time
import os
import math
import time
import sys

import argparse
import logging
import numpy as np
import tensorflow.contrib.slim as slim
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from enum import Enum
import cv2 as cv

from tf_qrnn import QRNN
from data_cmp_generate import get_evaldata, write_mat, write_mat_conf, get_relevant_prediction_index
#from generate_low_res_facade import next_batch, write_mat, get_relevant_prediction_index

from md_lstm import *
from own_loss_test import shape_loss_batch, distbce_loss_batch
from mdmd_lstm import multi_directional_md_rnn_while_loop
from match_img import list_data, match_matrix

THRESHOLD = 0.2
#PATH_TEMPLATE = "/media/DATA/simon/mdlstm_modified/tensorflow-multi-dimensional-lstm/"
BEGINS_WITH = "checkpoint"

OUTPUT_GRID = "outputs/out_graz_evalgrid25/"
OUTPUT_QRNN = "outputs/out_graz_evalqrnn25/"
OUTPUT_MDLSTM = "outputs/out_graz_evalmdlstm25/"
OUTPUT_MDMDLSTM = "outputs/out_graz_evalmdmdlstm25/"
OUTPUT_MDLSTM_QRNN1 = "outputs/out_graz_evalmdlstmqrnn1_25/"
OUTPUT_MDLSTM_QRNN2 = "outputs/out_graz_evalmdlstmqrnn2_25/"

GRAZ_ORIGIN = "/media/DATA/simon/Repos/facadecompletion/data/graz50/graz50_matrix/"

#logger = logging.getLogger(__name__)

#calc TP
"""
def calc_tp( prediction, gt, color):
    #print(prediction.shape == gt.shape)
    if (prediction.shape == gt.shape) :
        width = prediction.shape[0]
        height = prediction.shape[1]
        #all_pixel = width*height
        tp = 0.0

        for x in range(width):
            for y in range(height):
                if  (gt[x][y] == color).all()  and (prediction[x][y] == color).all():
                    tp += 1.0
        return tp

    else:
        print("ERROR: Prediction and GT not same shape!!!")
        return -1
"""

def list_checkpoint_dirs(input_path, template):
    #print("PATH %s" % input_path)
    #input_path = input_path+"/"
    begins_with='0'
    image_list = []
    file_list = os.listdir(input_path) # Change this PATH to traverse other directories if you want.
    if file_list != None:
        pass
    #print("%s files were found under current folder. " % len(file_list))
    #print("Please be noted that only files end with '*.jpg' will be load!")
    for i in range(len(file_list)):
        current_file_abs_path = input_path+file_list[i]
        if (current_file_abs_path.startswith(template)):
            image_list.append(current_file_abs_path)
    image_list.sort()
    #print("#####################################")
    #print(len(image_list))
    #print("#####################################")
    return image_list

def calc_tp( prediction, gt, threshold):
    if (prediction.shape == gt.shape):
        width = prediction.shape[0]
        height = prediction.shape[1]
        #all_pixel = width*height
        tp = 0.0

        for x in range(width):
            for y in range(height):
                if  (gt[x][y] >= threshold)  and (prediction[x][y] >= threshold):
                    tp += 1.0
        return tp

    else:
        print("ERROR: Prediction and GT not same shape!!!")
        return -1

#calc FP
def calc_fp( prediction, gt, threshold):
    if (prediction.shape == gt.shape):
        width = prediction.shape[0]
        height = prediction.shape[1]
        #all_pixel = width*height
        fp = 0.0

        for x in range(width):
            for y in range(height):
                if (gt[x][y] < threshold) and (prediction[x][y] >= threshold):
                    fp += 1.0
        return fp


    else:
        print("ERROR: Prediction and GT not same shape!!!")
        return -1

#calc TN
def calc_tn(prediction, gt, threshold):
    if (prediction.shape == gt.shape):
        width = prediction.shape[0]
        height = prediction.shape[1]
        #all_pixel = width*height
        tn = 0.0

        for x in range(width):
            for y in range(height):
                if (gt[x][y] < threshold) and (prediction[x][y] < threshold):
                    tn += 1.0
        return tn


    else:
        print("ERROR: Prediction and GT not same shape!!!")
        return -1

#calc FN
def calc_fn( prediction, gt, threshold):
    if (prediction.shape == gt.shape):
        width = prediction.shape[0]
        height = prediction.shape[1]
        #all_pixel = width*height
        fn = 0.0

        for x in range(width):
            for y in range(height):
                if (gt[x][y] >= threshold) and (prediction[x][y] < threshold):
                    fn += 1.0
        return fn


    else:
        print(prediction.shape)
        print(gt.shape)
        print("ERROR: Prediction and GT not same shape!!!")
        return -1

#Calc Accuracy for one class (color)
def calc_evalvalues( prediction, gt, threshold):
    tp = calc_tp(prediction, gt, threshold)
    fp = calc_fp(prediction, gt, threshold)
    tn = calc_tn(prediction, gt, threshold)
    fn = calc_fn(prediction, gt, threshold)
    total = prediction.shape[0]*prediction.shape[1]
    #print("Total Pixel: "+ str(total))
    #print("TP: "+ str(tp))
    #print("FP: "+ str(fp))
    #print("TN: "+ str(tn))
    #print("FN: "+ str(fn))

    if (tp+tn+fp+fn) != total:
        print("ERROR: "+str(total)+" =/= "+str(tp+tn+fp+fn))

    if tp == 0 and fp == 0 and tn == 0 and fn == 0:
        acc = -1.0
    else:
        acc = (tp + tn)/(tp+tn+fp+fn)

    if tp == 0 and fp == 0:
        pre = -1.0
    else:
        pre = (tp)/(tp+fp)

    if tp == 0 and fn == 0:
        recall = -1.0
    else:
        recall = (tp)/(tp+fn)

    if tp == 0 and fp == 0 and fn == 0:
        iou = -1.0
    else:
        iou = (tp)/(tp+fp+fn)


    return acc, pre, recall, iou

#Calc Accuracy for one class (color)
def calc_precision( prediction, gt, color):
    tp = calc_tp(prediction, gt, color)
    fp = calc_fp(prediction, gt, color)
    tn = calc_tn(prediction, gt, color)
    fn = calc_fn(prediction, gt, color)

    if tp == 0 and fp == 0:
        pre = -1.0
    else:
        pre = (tp)/(tp+fp)

    return pre

#Calc Accuracy for one class (color)
def calc_recall( prediction, gt, color):
    tp = calc_tp(prediction, gt, color)
    fp = calc_fp(prediction, gt, color)
    tn = calc_tn(prediction, gt, color)
    fn = calc_fn(prediction, gt, color)

    if tp == 0 and fn == 0:
        recall = -1.0
    else:
        recall = (tp)/(tp+fn)

    return recall

def calc_IoU(prediction, target):
    #print(target.shape)
    #print(prediction.shape)
    #cv.imshow('image',prediction)
    #cv.waitKey(0)
    #cv.destroyAllWindows()
    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    iou_score = np.sum(intersection) / np.sum(union)

    return iou_score

def write_detections(filepath,output,groundtruth,input,threshold):
    image_list = list_data(GRAZ_ORIGIN, "_facade.png")
    spatial_list = list_data(GRAZ_ORIGIN, "_spatial.txt")

    #print("Spatial: "+str(spatial_list))

    index = match_matrix(groundtruth)

    def read_spatial(spatial_filepath):
        spatial_file = open(spatial_filepath, "r")

        line_x = spatial_file.readline()
        line_y = spatial_file.readline()

        grid_x = line_x.split(" ")
        grid_y = line_y.split(" ")

        int_x = []
        int_y = []

        #print(grid_x)
        #print(grid_y)

        if grid_x[0] == 'x' and grid_y[0] == 'y':
            for x in range(1,len(grid_x)):
                if grid_x[x]:
                    int_x.append(int(grid_x[x].rstrip("\n")))

            for y in range(1,len(grid_y)):
                if grid_y[y]:
                    int_y.append(int(grid_y[y].rstrip("\n")))
        else:
            print("ERROR: File Corrupted!")
            exit()

        #print(int_x)
        #print(int_y)

        return int_x, int_y

    grid_x, grid_y = read_spatial(spatial_list[index])

    img = cv.imread(image_list[index])

    def draw_box(img, color, x_min, y_min, x_max, y_max):
        # Line thickness of 9 px
        thickness = 2

        # Using cv2.line() method
        # Draw a diagonal green line with thickness of 9 px
        img = cv.line(img, (x_min,y_min), (x_max,y_min), color, thickness)
        img = cv.line(img, (x_min,y_min), (x_min,y_max), color, thickness)

        img = cv.line(img, (x_max,y_max), (x_max,y_min), color, thickness)
        img = cv.line(img, (x_max,y_max), (x_min,y_max), color, thickness)

        return img

    y_len = len(grid_y)-1#output.shape[0] #CHANGED SWITCH
    x_len = len(grid_x)-1#output.shape[1]
    #print("\n")

    #print(y_len)
    #print(x_len)

    #print(len(grid_x))
    #print(len(grid_y))
    def write_between(img,y,x,filepath):
        between = "_"+str(y)+"_"+str(x)

        splits = filepath.split('.')


        new_path = splits[0]+between+"."+splits[1]
        print(new_path)

        cv.imwrite(new_path,img)

    for y in range(y_len):
        for x in range(x_len):
            if output[y][x] > threshold:
                # Green color in BGR
                color = (0, 255, 0)
                if input[y][x] > 0.5:
                    color = (0,0,255)

                #print(x)
                #print(y)

                #print("")
                x_min = grid_x[x]
                x_max = grid_x[x+1]

                y_min = grid_y[y]
                y_max = grid_y[y+1]

                img = draw_box(img, color, x_min, y_min, x_max, y_max)
                #write_between(img,y,x,filepath)

    cv.imwrite(filepath,img)

    return img


def get_script_arguments():
    parser = argparse.ArgumentParser(description='MD LSTM trainer.')
    parser.add_argument('--model_type', required=True, type=ModelType.from_string,
                        choices=list(ModelType), help='Model type.')
    parser.add_argument('--enable_plotting', action='store_true')

    parser.add_argument('--checkpoint_path', required=True, help='Path to ckeckpoint to use.')

    parser.add_argument('--hidden_size', help='Hiddensize')

    args = get_arguments(parser)
    #logger.info('Script inputs: {}.'.format(args))
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
    DQRNN = 'DQRNN'
    MD_QRNN_COMBI = 'MD_QRNN_COMBI'
    MD_QRNN_COMBI2 = 'MD_QRNN_COMBI2'

    def __str__(self):
        return self.value

    @staticmethod
    def from_string(s):
        try:
            return ModelType[s]
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


def eval(model_type='mdmd_lstm', enable_plotting=True, checkpoint_path='checkpoint', hidden_size=128):
    learning_rate = 0.01
    batch_size = 1
    in_h = 25#64#32#128
    in_w = 25#64#32#128
    out_h = 25#64#32#128
    out_w = 25#64#32#128
    channels = 1
    #hidden_size = 128#64#256

    #x = tf.placeholder(tf.float32, [batch_size, h, w, channels])
    #y = tf.placeholder(tf.float32, [batch_size, h, w, channels])
    x = tf.placeholder(tf.float32, [batch_size, in_h, in_w, channels])

    if model_type == ModelType.MD_LSTM:
        print(model_type)
        y = tf.placeholder(tf.float32, [batch_size, out_h, out_w, channels])
        #logger.info('Using Multi Dimensional LSTM.')
        rnn_out, _ = multi_dimensional_rnn_while_loop(rnn_size=hidden_size, input_data=x, sh=[1, 1])
    elif model_type == ModelType.MDMD_LSTM:
        #hidden_size = 128#64#256
        print(model_type)
        y = tf.placeholder(tf.float32, [batch_size, out_h, out_w, channels])
        #logger.info('Using Multi Dimensional LSTM.')
        rnn_out, _ = multi_directional_md_rnn_while_loop(rnn_size=hidden_size, input_data=x, sh=[1, 1])
    elif model_type == ModelType.HORIZONTAL_SD_LSTM:
        print(model_type)
        y = tf.placeholder(tf.float32, [batch_size, out_h, out_w, channels])
        #logger.info('Using Standard LSTM.')
        rnn_out = horizontal_standard_lstm(input_data=x, rnn_size=hidden_size)
    elif model_type == ModelType.SNAKE_SD_LSTM:
        print(model_type)
        y = tf.placeholder(tf.float32, [batch_size, out_h, out_w, channels])
        rnn_out = snake_standard_lstm(input_data=x, rnn_size=hidden_size)
    elif model_type == ModelType.SNAKE_GRID_LSTM:
        print("???????????????????????????????????????????????????????????????")
        #hidden_size = 256#64#256
        print(model_type)
        y = tf.placeholder(tf.float32, [batch_size, out_h, out_w, channels])
        rnn_out = snake_grid_lstm(input_data=x, rnn_size=hidden_size)
    elif model_type == ModelType.MD_SNAKE_GRID_LSTM:
        #hidden_size = 256#64#256
        print(model_type)
        y = tf.placeholder(tf.float32, [batch_size, out_h, out_w, channels])
        rnn_out = md_snake_grid_lstm(input_data=x, rnn_size=hidden_size)
    elif model_type == ModelType.MD_SNAKE_GRID_LSTM_C:
        print(model_type)

        learning_rate = 0.01
        batch_size = 16
        in_h = 64#32#128
        in_w = 64#32#128
        #out_h = 16#32#128
        out_len = 64#32#128
        max_pool_h = int(math.sqrt(in_h))
        max_pool_w = int(math.sqrt(in_w))
        channels = 1
        #hidden_size = 256#128#64#256
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
        #hidden_size = 256#128#64#256
        x = tf.placeholder(tf.float32, [batch_size, input_size, input_size, channels])
        y = tf.placeholder(tf.float32, [batch_size, out_len, out_len, channels])
        rnn_out = resnet(
            input_data=x,
            resnet_size=hidden_size,
            input_size=input_size,
            training=True
        )
    elif model_type == ModelType.MD_LSTM_DISTANCE:
        print(model_type)
        y = tf.placeholder(tf.float32, [batch_size, out_h, out_w, channels])
        #logger.info('Using Multi Dimensional LSTM.')
        rnn_out, _ = multi_dimensional_rnn_while_loop(rnn_size=hidden_size, input_data=x, sh=[1, 1])
    elif model_type == ModelType.MDMD_LSTM_DISTANCE:
        print(model_type)
        y = tf.placeholder(tf.float32, [batch_size, out_h, out_w, channels])
        #y_dist1 = tf.placeholder(tf.float32, [batch_size, out_h, out_w, channels])
        #y_dist2 = tf.placeholder(tf.float32, [batch_size, out_h, out_w, channels])
        rnn_out, _ = multi_directional_md_rnn_while_loop(input_data=x, rnn_size=hidden_size,sh=[1, 1])
    elif model_type == ModelType.QRNN:
        y = tf.placeholder(tf.float32, [batch_size, out_h, out_w, channels])
        #logger.info('Using QRNN.')
        size = in_h
        in_size = in_h
        conv_size = hidden_size
        qrnn = QRNN(b_size=batch_size,in_size=1, size=size*in_size, conv_size=conv_size)
        #qrnn = QRNN(in_size=word_size, size=size, conv_size=3)
        #x_reshaped = tf.squeeze(tf.reshape(x,[batch_size, in_w*in_h, channels]))
        x_reshaped = tf.reshape(x,[batch_size, in_w*in_h, channels])
        #x_reshaped = tf.squeeze(x_reshaped)
        print(x_reshaped)
        rnn_out = qrnn.forward(x_reshaped)
        rnn_out = tf.reshape(rnn_out,[batch_size, in_w, in_h, channels])
        print("rnn_out:")
        print(rnn_out)
        '''
        with tf.name_scope("QRNN-Classifier"):
            W = tf.Variable(tf.random_normal([size, size]), name="W")
            b = tf.Variable(tf.random_normal([size]), name="b")
            rnn_out = tf.add(tf.matmul(hidden, W), b)
        #in_size=in_size, size=size, conv_size=conv_size
        '''
    elif model_type == ModelType.MD_QRNN_COMBI:
        print(model_type)
        size = in_h
        in_size = channels
        y = tf.placeholder(tf.float32, [batch_size, out_h, out_w, channels])
        #logger.info('Using Multi Dimensional LSTM.')
        rnn_out = md_qrnn_combi(input_data=x, rnn_size=hidden_size, size=size, in_size=in_size, batch_size=batch_size)
    elif model_type == ModelType.MD_QRNN_COMBI2:
        print(model_type)
        size = in_h
        in_size = channels
        y = tf.placeholder(tf.float32, [batch_size, out_h, out_w, channels])
        #logger.info('Using Multi Dimensional LSTM.')
        rnn_out = md_qrnn_combi2(input_data=x, rnn_size=hidden_size, size=size, in_size=in_size, batch_size=batch_size)
    else:
        raise Exception('Unknown model type: {}.'.format(model_type))
    #GRADIENT CLIPPING
    """
    loss = tf.reduce_mean(tf.square(y - model_out))
    grad_update = tf.train.AdamOptimizer(learning_rate)#.minimize(loss)
    gvs = grad_update.compute_gradients(loss)
    capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
    train_op = grad_update.apply_gradients(capped_gvs)
    gpu_options = tf.GPUOptions(allow_growth = True)

    loss = tf.reduce_mean(tf.square(y - model_out))
    grad_update = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    gpu_options = tf.GPUOptions(allow_growth = True)
    # Add ops to save and restore all the variables.

    """
    print(model_type)

    if not (model_type == ModelType.QRNN or model_type == ModelType.MD_QRNN_COMBI or model_type == ModelType.DQRNN):
        print("SOLLTE NICHT SEIN")
        model_out = slim.fully_connected(inputs=rnn_out,
                                     num_outputs=1,
                                     activation_fn=tf.nn.sigmoid)
    else:
        model_out = rnn_out

    print("pool_out: "+str(model_out))
    reshape_out = tf.reshape(model_out, y.shape)
    print("reshape_out: "+str(reshape_out))


    gpu_options = tf.GPUOptions(allow_growth = True)
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False,gpu_options=gpu_options))
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    # Restore variables from disk.
    #if os.path.isfile("checkpoint/model.ckpt"):

    print("##############################")
    #print_tensors_in_checkpoint_file(PATH_TEMPLATE+checkpoint_path, all_tensors=True, tensor_name='')
    print("##############################")
    print("Trying to use "+checkpoint_path)
    print("##############################")
    saver.restore(sess, checkpoint_path)
    print("Model restored.")

    #fp = FileLogger('out_{}.tsv'.format(model_type), ['steps_{}'.format(model_type),
    #                                                  'overall_loss_{}'.format(model_type),
    #                                                  'time_{}'.format(model_type),
    #                                                  'relevant_loss_{}'.format(model_type)])
    input_data, gt_data = get_evaldata()

    #print("DEBUG INPU_DATA: "+str(input_data.shape))
    if len(input_data) == 0:
        print("INPUT_DATA EMPTY")
    if len(gt_data) == 0:
        print("GT_DATA EMPTY")
    results = []
    thresholds = []
    avg_prec_list = []
    avg_reca_list = []
    avg_acur_list = []
    avg_iou_list = []
    toolbar_width = 50
    step_mod = int((len(input_data))/toolbar_width)


    print("START VALUES...")
    all_acc = []
    all_pre = []
    all_recall = []
    all_iou = []

    # setup toolbar
    sys.stdout.write("[%s]" % (" " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['

    for i in range(len(input_data)):
        if (i-1) % step_mod == 0:
            # update the bar
            sys.stdout.write("-")
            sys.stdout.flush()



        tmp_acc, tmp_pre, tmp_recall, tmp_iou = calc_evalvalues(np.squeeze(np.array(input_data[i])),np.squeeze(np.array(gt_data[i])), 0.5)
        all_acc.append(tmp_acc)
        all_pre.append(tmp_pre)
        all_recall.append(tmp_recall)
        all_iou.append(tmp_iou)
    sys.stdout.write("\n")

    avg_acc = 0.0
    avg_pre = 0.0
    avg_recall = 0.0
    avg_iou = 0.0

    for i in range(len(all_acc)):
        #print("###############"+str(i)+"################")
        #print("all_acc[i]:"+str(all_acc[i]))
        #print("all_pre[i]:"+str(all_pre[i]))
        #print("all_recall[i]:"+str(all_recall[i]))
        #print("all_iou[i]:"+str(all_iou[i]))
        avg_acc += all_acc[i]
        avg_pre += all_pre[i]
        avg_recall += all_recall[i]
        avg_iou += all_iou[i]

    avg_acc /= len(all_acc)
    avg_pre /= len(all_pre)
    avg_recall /= len(all_pre)
    avg_iou /= len(all_pre)

    print("#########START##########")
    print("Avg. Accuracy: {:01.6}".format(avg_acc))
    print("Avg. Precision: {:01.6}".format(avg_pre))
    print("Avg. Recall: {:01.6}".format(avg_recall))
    print("Avg. Intersection over Union: {:01.6}".format(avg_iou))
    print("###################")

    print(" Getting Predictions...")
    # setup toolbar
    sys.stdout.write("[%s]" % (" " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['


    results = []

    for i in range(len(input_data)):
        batch = []
        batch.append(input_data[i])
        #batch = next_batch(batch_size)
        #grad_step_start_time = time()
        if (i-1) % step_mod == 0:
            # update the bar
            sys.stdout.write("-")
            sys.stdout.flush()

        tmp_x = np.array(batch)
        batch_x = np.expand_dims(tmp_x, axis=3)
        #batch_y = np.expand_dims(batch[1], axis=3)

        model_preds = sess.run([reshape_out], feed_dict={x: batch_x})
        mat_out = np.array(model_preds)
        mat_gt = np.array(gt_data[i])

        if i < 50:
            if model_type == ModelType.QRNN:
                x_name = "x_"+'{:06d}'.format(i)+".png"
                y_name = "y_"+'{:06d}'.format(i)+".png"
                z_name = "z_"+'{:06d}'.format(i)+".png"
                #print("\n")
                #print(batch_x.shape)
                #print(mat_out.shape)
                #print(mat_gt.shape)
                #print("\n")
                write_mat_conf(batch_x.squeeze(), x_name, OUTPUT_QRNN)
                write_mat_conf(mat_out.squeeze(), z_name, OUTPUT_QRNN)
                write_mat_conf(mat_gt.squeeze(), y_name, OUTPUT_QRNN)
            elif model_type == ModelType.SNAKE_GRID_LSTM:
                x_name = "x_"+'{:06d}'.format(i)+".png"
                y_name = "y_"+'{:06d}'.format(i)+".png"
                z_name = "z_"+'{:06d}'.format(i)+".png"

                #print("\n")
                #print(batch_x.shape)
                #print(mat_out.shape)
                #print(mat_gt.shape)
                #print("\n")

                write_mat_conf(batch_x.squeeze(), x_name, OUTPUT_GRID)
                write_mat_conf(mat_out.squeeze(), z_name, OUTPUT_GRID)
                write_mat_conf(mat_gt.squeeze(), y_name, OUTPUT_GRID)
            elif model_type == ModelType.MD_LSTM:
                x_name = "x_"+'{:06d}'.format(i)+".png"
                y_name = "y_"+'{:06d}'.format(i)+".png"
                z_name = "z_"+'{:06d}'.format(i)+".png"

                #print("\n")
                #print(batch_x.shape)
                #print(mat_out.shape)
                #print(mat_gt.shape)
                #print("\n")

                write_mat_conf(batch_x.squeeze(), x_name, OUTPUT_MDLSTM)
                write_mat_conf(mat_out.squeeze(), z_name, OUTPUT_MDLSTM)
                write_mat_conf(mat_gt.squeeze(), y_name, OUTPUT_MDLSTM)
            elif model_type == ModelType.MDMD_LSTM:
                x_name = "x_"+'{:06d}'.format(i)+".png"
                y_name = "y_"+'{:06d}'.format(i)+".png"
                z_name = "z_"+'{:06d}'.format(i)+".png"
                det_name = "det_"+'{:06d}'.format(i)+".png"

                #print("\n")
                #print(batch_x.shape)
                #print(mat_out.shape)
                #print(mat_gt.shape)
                #print("\n")

                write_mat_conf(batch_x.squeeze(), x_name, OUTPUT_MDMDLSTM)
                write_mat_conf(mat_out.squeeze(), z_name, OUTPUT_MDMDLSTM)
                write_mat_conf(mat_gt.squeeze(), y_name, OUTPUT_MDMDLSTM)
                write_detections(OUTPUT_MDMDLSTM+det_name,mat_out.squeeze(),mat_gt.squeeze(), batch_x.squeeze(), 0.5)
            elif model_type == ModelType.MD_QRNN_COMBI:
                x_name = "x_"+'{:06d}'.format(i)+".png"
                y_name = "y_"+'{:06d}'.format(i)+".png"
                z_name = "z_"+'{:06d}'.format(i)+".png"

                #print("\n")
                #print(batch_x.shape)
                #print(mat_out.shape)
                #print(mat_gt.shape)
                #print("\n")

                write_mat_conf(batch_x.squeeze(), x_name, OUTPUT_MDLSTM_QRNN1)
                write_mat_conf(mat_out.squeeze(), z_name, OUTPUT_MDLSTM_QRNN1)
                write_mat_conf(mat_gt.squeeze(), y_name, OUTPUT_MDLSTM_QRNN1)
            elif model_type == ModelType.MD_QRNN_COMBI2:
                x_name = "x_"+'{:06d}'.format(i)+".png"
                y_name = "y_"+'{:06d}'.format(i)+".png"
                z_name = "z_"+'{:06d}'.format(i)+".png"

                #print("\n")
                #print(batch_x.shape)
                #print(mat_out.shape)
                #print(mat_gt.shape)
                #print("\n")

                write_mat_conf(batch_x.squeeze(), x_name, OUTPUT_MDLSTM_QRNN2)
                write_mat_conf(mat_out.squeeze(), z_name, OUTPUT_MDLSTM_QRNN2)
                write_mat_conf(mat_gt.squeeze(), y_name, OUTPUT_MDLSTM_QRNN2)
            else:
                pass
        #model_preds, tot_loss_value, _ = sess.run([model_out, loss, grad_update], feed_dict={x: batch_x, y: batch_y}
        results.append(model_preds)
    sys.stdout.write("\n")



    for threshold in np.arange(0.1,0.9,0.1):
        thresholds.append(threshold)

        step_mod = int((len(input_data))/toolbar_width)


        print(str(threshold)+" Calculate Evaluation...")
        all_acc = []
        all_pre = []
        all_recall = []
        all_iou = []

        # setup toolbar
        sys.stdout.write("[%s]" % (" " * toolbar_width))
        sys.stdout.flush()
        sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['

        for i in range(len(results)):
            if (i-1) % step_mod == 0:
                # update the bar
                sys.stdout.write("-")
                sys.stdout.flush()

            tmp_acc, tmp_pre, tmp_recall, tmp_iou = calc_evalvalues( np.squeeze(np.array(results[i])),np.squeeze(np.array(gt_data[i])), threshold)
            all_acc.append(tmp_acc)
            all_pre.append(tmp_pre)
            all_recall.append(tmp_recall)
            all_iou.append(tmp_iou)
        sys.stdout.write("\n")

        avg_acc = 0.0
        avg_pre = 0.0
        avg_recall = 0.0
        avg_iou = 0.0

        for i in range(len(all_acc)):
            avg_acc += all_acc[i]
            avg_pre += all_pre[i]
            avg_recall += all_recall[i]
            avg_iou += all_iou[i]

        avg_acc /= len(all_acc)
        avg_pre /= len(all_pre)
        avg_recall /= len(all_pre)
        avg_iou /= len(all_pre)

        avg_acur_list.append(avg_acc)
        avg_prec_list.append(avg_pre)
        avg_reca_list.append(avg_recall)
        avg_iou_list.append(avg_iou)

        print("#########"+str(threshold)+"##########")
        print("Avg. Accuracy: {:01.6}".format(avg_acc))
        print("Avg. Precision: {:01.6}".format(avg_pre))
        print("Avg. Recall: {:01.6}".format(avg_recall))
        print("Avg. Intersection over Union: {:01.6}".format(avg_iou))
        print("###################")

    for i in range(len(thresholds)):
        print("#########"+str(thresholds[i])+"##########")
        print("Avg. Accuracy: {:01.6}".format(avg_acur_list[i]))
        print("Avg. Precision: {:01.6}".format(avg_prec_list[i]))
        print("Avg. Recall: {:01.6}".format(avg_reca_list[i]))
        print("Avg. Intersection over Union: {:01.6}".format(avg_iou_list[i]))
        print("###################")

def main():
    args = get_script_arguments()
    print(args.hidden_size)
    logging.basicConfig(format='%(asctime)12s - %(levelname)s - %(message)s', level=logging.INFO)
    eval(args.model_type, args.enable_plotting, args.checkpoint_path, int(args.hidden_size))


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    main()
