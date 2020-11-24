from data_cmp_generate import *
from md_lstm import *
from own_loss_test import shape_loss_batch, distbce_loss_batch
from mdmd_lstm import multi_directional_md_rnn_while_loop
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
from tf_qrnn import QRNN
from cv2 import cv2
from generate_dataset import *#parse_xml, cluster_gridlines, prepare_inputdata, extract_grids, list_xml, list_imgs

"""
This should be the main Programm, this should be done:
- read detections and parse them into 2d arrays
- use a choosen LSTM to give proposals
- return new detections
Arguments:
- Choosen Network (maybe if no checkpoint available train the network)
- Threshold for Proposals (default 5)
"""
###########################################################################################################
def get_arguments(parser: argparse.ArgumentParser):
    args = None
    try:
        args = parser.parse_args()
    except Exception:
        parser.print_help()
        exit(1)
    return args

def get_script_arguments():
    parser = argparse.ArgumentParser(description='MD LSTM trainer.')
    parser.add_argument('--model_type', required=True, type=ModelType.from_string,
                        choices=list(ModelType), help='Model type.')
    parser.add_argument('--enable_plotting', action='store_true')

    parser.add_argument('--threshold', help='threshold for teh proposals')

    parser.add_argument('--max_size', help='maximum size of input')

    parser.add_argument('--detections_dir', help='detections_dir Path')

    parser.add_argument('--image_dir', help='image_dir')

    parser.add_argument('--output_dir', help='output_dir')
                    
    args = get_arguments(parser)
    #logger.info('Script inputs: {}.'.format(args))
    return args

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

def split_facade(matrix): #TODO
    #Trying to split facade into two, to process it
    #middle = max(matrix.shape[0], matrix.shape[1])
    new_arrays = []

    if matrix.shape[0] > matrix.shape[1]:
        middle = int(matrix.shape[0]/2)

        while np.all(matrix[middle,:] == 0.0):
            middle +=1
        tmp_array = np.copy(matrix[:middle,:])#np.split(matrix,2,0)
        new_arrays.append(tmp_array)
        tmp_array = np.copy(matrix[middle-1:,:])#np.split(matrix,2,0)
        new_arrays.append(tmp_array)

    else:
        middle = int(matrix.shape[1]/2)

        while np.all(matrix[:,middle] == 0.0):
            middle +=1
        tmp_array = np.copy(matrix[:,:middle])#np.split(matrix,2,0)
        new_arrays.append(tmp_array)
        tmp_array = np.copy(matrix[:,middle-1:])#np.split(matrix,2,0)
        new_arrays.append(tmp_array)
    
    return new_arrays

def zero_padin(matrix, size_p):
    matrix = np.array(matrix)
    print(matrix.shape)
    new_matrix = np.random.uniform(low=0.00001, high=0.001, size=(size_p,size_p))
    #print(matrix)

    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            new_matrix[y][x] = matrix[y][x]

    return new_matrix





#########################################################################################################

def read_detections(detection_path, image_path, max_size):
    #print(detection_path)
    #LOAD ALL XML FILES
    all_bboxes = []
    all_xml = list_xml(detection_path)
    #print(len(all_xml))
    for i in range(len(all_xml)):
        bboxes = parse_xml(all_xml[i])
        all_bboxes.append(cluster_gridlines(bboxes))
        #print("Objects: "+str(len(bboxes)))
    #LOAD ALL IMAGES
    all_images = list_imgs(image_path)
    images_loaded = []
    for i in range(len(all_images)):
        #print("#######################")
        #print(all_xml[i])
        #print("-----------------------")
        #print(all_images[i])
        #print("#######################")
        tmp_img = cv2.imread(all_images[i])
        images_loaded.append(tmp_img.copy())

    #CONSTRUCT GRIDS
    all_grids = extract_grids(all_bboxes, images_loaded)
    new_all_grids = []

    #CLUSTER GRIDS
    for i in range(len(all_grids)):
        new_all_grids.append(cluster_gridlines(all_grids[i]))

    #WRITE MATRIX
    all_matrixes = []
    for i in range(len(all_bboxes)):
            #Data needed for further work
            img = images_loaded[i]
            boxes = all_bboxes[i]
            x_grid = all_grids[i][0]
            y_grid = all_grids[i][1]
            #print(np.array(all_grids).shape)
            #print("##############################")
            #print("IMG SHAPE: "+str(img.shape))
            #print(len(x_grid))
            #print("MAX XGRID: "+str(x_grid[len(x_grid)-1]))
            #print(len(y_grid))
            #print("MAX YGRID: "+str(y_grid[len(y_grid)-1]))
            #print("##############################")

            #construct empty matrix
            matrix = np.zeros((len(y_grid)-1,len(x_grid)-1))

            #fill matrix with 1 = detected object
            #print(matrix)
            #print(x_grid)
            #print(y_grid)
            #print(boxes)

            matrix = set_object(matrix, x_grid, y_grid, boxes)
            
            all_matrixes.append(matrix)

    print("Total befor Split: "+str(len(all_matrixes)))

    #Split TO BIG
    found_too_big = 1
    while found_too_big == 1:
        found_too_big = 0
        new_arrangement = []
        for i in range(len(all_matrixes)):
            if all_matrixes[i].shape[0] > max_size or all_matrixes[i].shape[1] > max_size:
                found_too_big = 1
                tmp_matrix = all_matrixes[i]
                print("WARNING: "+str(tmp_matrix.shape)+" is too big. MAX Size 25x25")
                print("Trying to split")
                new_matrixes = split_facade(tmp_matrix)
                print("new matrixes shapes:")
                print(new_matrixes[0].shape)
                print(new_matrixes[1].shape)
                
                new_arrangement.append(np.copy(new_matrixes[0]))
                new_arrangement.append(np.copy(new_matrixes[1]))
            else:
                new_arrangement.append(all_matrixes[i])

        all_matrixes = new_arrangement


    #Zero Padding recoverable
    for i in range(len(all_matrixes)):
        all_matrixes[i] = zero_padin(all_matrixes[i], max_size)

    print("Total: "+str(len(all_matrixes)))
    if len(all_matrixes) == 0:
        exit()
    
    return all_matrixes


    #GENERATE TRAININGSDATA USING GROUNDTRUTH
    #print("all_grids: "+str(np.array(all_grids).shape))
    #print("new_all_grids: "+str(np.array(new_all_grids).shape))
    #prepared_td = prepare_inputdata(all_bboxes, images_loaded, all_grids)
    #prepared_td = prepare_inputdata(all_bboxes, images_loaded, new_all_grids)
    #prepared_gt = prepare_gtdata(bboxes, images_loaded, all_grids)

def use_network(detections, args):
    model_type= args.model_type
    enable_plotting = args.enable_plotting

    if model_type == ModelType.MD_LSTM:
        checkpoint_path= 'checkpoints/checkpoint_mdlstm/model.ckpt'
    elif model_type == ModelType.MDMD_LSTM:
        checkpoint_path= 'checkpoints/checkpoint_mdmdlstm/model.ckpt'
    elif model_type == ModelType.QRNN:
        checkpoint_path= 'checkpoints/checkpoint_qrnn/model.ckpt'

    batch_size = 1
    in_h = 25
    in_w = 25
    out_h = 25
    out_w = 25
    channels = 1

    x = tf.placeholder(tf.float32, [batch_size, in_h, in_w, channels])

    if model_type == ModelType.MD_LSTM:
        hidden_size = 2500#4096#1024#2048#4096#256#128#64#256 #2500 for size 25
        print(model_type)
        y = tf.placeholder(tf.float32, [batch_size, out_h, out_w, channels])
        #logger.info('Using Multi Dimensional LSTM.')
        rnn_out, _ = multi_dimensional_rnn_while_loop(rnn_size=hidden_size, input_data=x, sh=[1, 1])
    elif model_type == ModelType.MDMD_LSTM:
        #hidden_size = 128#64#256
        hidden_size = 256
        print(model_type)
        y = tf.placeholder(tf.float32, [batch_size, out_h, out_w, channels])
        #logger.info('Using Multi Dimensional LSTM.')
        rnn_out, _ = multi_directional_md_rnn_while_loop(rnn_size=hidden_size, input_data=x, sh=[1, 1])
    elif model_type == ModelType.HORIZONTAL_SD_LSTM:
        print(model_type)
        hidden_size = 256
        y = tf.placeholder(tf.float32, [batch_size, out_h, out_w, channels])
        #logger.info('Using Standard LSTM.')
        rnn_out = horizontal_standard_lstm(input_data=x, rnn_size=hidden_size)
    elif model_type == ModelType.SNAKE_SD_LSTM:
        hidden_size = 256
        print(model_type)
        y = tf.placeholder(tf.float32, [batch_size, out_h, out_w, channels])
        rnn_out = snake_standard_lstm(input_data=x, rnn_size=hidden_size)
    elif model_type == ModelType.SNAKE_GRID_LSTM:
        hidden_size = 256
        print("???????????????????????????????????????????????????????????????")
        #hidden_size = 256#64#256
        print(model_type)
        y = tf.placeholder(tf.float32, [batch_size, out_h, out_w, channels])
        rnn_out = snake_grid_lstm(input_data=x, rnn_size=hidden_size)
    elif model_type == ModelType.MD_SNAKE_GRID_LSTM:
        hidden_size = 256
        #hidden_size = 256#64#256
        print(model_type)
        y = tf.placeholder(tf.float32, [batch_size, out_h, out_w, channels])
        rnn_out = md_snake_grid_lstm(input_data=x, rnn_size=hidden_size)
    elif model_type == ModelType.MD_SNAKE_GRID_LSTM_C:
        hidden_size = 256
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
        hidden_size = 256
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
        hidden_size = 256
        print(model_type)
        y = tf.placeholder(tf.float32, [batch_size, out_h, out_w, channels])
        #logger.info('Using Multi Dimensional LSTM.')
        rnn_out, _ = multi_dimensional_rnn_while_loop(rnn_size=hidden_size, input_data=x, sh=[1, 1])
    elif model_type == ModelType.MDMD_LSTM_DISTANCE:
        hidden_size = 256
        print(model_type)
        y = tf.placeholder(tf.float32, [batch_size, out_h, out_w, channels])
        #y_dist1 = tf.placeholder(tf.float32, [batch_size, out_h, out_w, channels])
        #y_dist2 = tf.placeholder(tf.float32, [batch_size, out_h, out_w, channels])
        rnn_out, _ = multi_directional_md_rnn_while_loop(input_data=x, rnn_size=hidden_size,sh=[1, 1])
    elif model_type == ModelType.QRNN:
        hidden_size = 128
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
        hidden_size = 256
        print(model_type)
        size = in_h
        in_size = channels
        y = tf.placeholder(tf.float32, [batch_size, out_h, out_w, channels])
        #logger.info('Using Multi Dimensional LSTM.')
        rnn_out = md_qrnn_combi(input_data=x, rnn_size=hidden_size, size=size, in_size=in_size, batch_size=batch_size)
    elif model_type == ModelType.MD_QRNN_COMBI2:
        hidden_size = 256
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

    print(" Getting Predictions...")
    # setup toolbar
    '''
    toolbar_width = 50
    step_mod = int((len(detections))/toolbar_width)
    sys.stdout.write("[%s]" % (" " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['
    '''

    results = []
    for i in range(len(detections)):
        batch = []
        batch.append(detections[i])
        #batch = next_batch(batch_size)
        #grad_step_start_time = time()
        #print(step_mod)
        #print(i)
        '''
        if step_mod == 0:
            step_mod=1

        if (i-1) % step_mod == 0:
            # update the bar
            sys.stdout.write("-")
            sys.stdout.flush()
        '''

        tmp_x = np.asarray(batch)
        batch_x = np.expand_dims(tmp_x, axis=3)
        #batch_y = np.expand_dims(batch[1], axis=3)

        #print(batch_x)

        model_preds = sess.run([reshape_out], feed_dict={x: batch_x})
        mat_out = np.array(model_preds)
        #mat_gt = np.array(gt_data[i])
        results.append(mat_out)

    return results

def write_proposals(new_detections, output_dir):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i in range(len(new_detections)):
        matrix = new_detections[i]

        test_file = "{0:06d}".format(i)+".txt"
        full_path = output_dir+test_file

        f = open(full_path, "w")

        for x in range(matrix.shape[0]):
            for y in range(matrix.shape[1]):
                f.write(str(matrix[x][y])+" ")
            f.write("\n")

        f.close()



if (__name__ == '__main__'):
    args = get_script_arguments()

    print("Reading Detections...")
    detections = read_detections(args.detections_dir, args.image_dir, int(args.max_size))

    print("Applying Neural Network...")
    new_detections = use_network(detections, args)

    print("Saving Proposals...")
    write_proposals(new_detections, args.output_dir)
