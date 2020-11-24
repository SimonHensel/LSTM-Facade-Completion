import numpy as np
import os
import re
#from scipy import interpolate
import cv2 as cv

from data_cmp_generate import parse_datafile, list_data, write_dist

PATH_DATA = "/media/DATA/simon/mdlstm_modified/tensorflow-multi-dimensional-lstm/data/tests_cluster/"
#PATH_DATA = "/home/simon/Dropbox/Arbeit/P6000/mdlstm_modified/tensorflow-multi-dimensional-lstm/data/tests/"
PATH_DISTANCES = "/media/DATA/simon/mdlstm_modified/tensorflow-multi-dimensional-lstm/data/Distances/"
PATH_OUTPUT = "/media/DATA/simon/mdlstm_modified/tensorflow-multi-dimensional-lstm/out_newloss/"
#PATH_OUTPUT = "/media/DATA/simon/mdlstm_modified/tensorflow-multi-dimensional-lstm/out_md/"
EVAL_PATH = "/media/DATA/simon/mdlstm_modified/tensorflow-multi-dimensional-lstm/eval/"

EVAL_SIZE = 10000
ZERO_PADDING_SIZE = 64#512
DIFICULTY = 8
DIFICULTY_PERCENTAGE = 0.3
DIFICULTY_MAX = 12
DIFICULTY_MIN = 8
GHOST_MIN = 0.4
GHOST_MAX = 0.6
MAX_FALSE = 3
MAX_FALSE_SIZE = 3
MAX_MISSING_COORDINATES = 32

def parse_datafile(file_to_read):

    lines = [line.rstrip('\n') for line in open(file_to_read)]

    #get sizes
    #print(file_to_read)
    rows = len(lines)
    cols = len(re.findall("\d+\.\d+", lines[0]))

    #print([int(s) for s in lines[0].split() if s.isdigit()])
    #cols = len([int(s) for s in lines[0].split() if s.isdigit()])
    #print("rows:"+str(rows))
    #print("cols:"+str(cols))

    #create numpy array
    #matrix = np.zeros((rows,cols),dtype=float)
    matrix = np.random.uniform(low=0.001, high=0.1, size=(rows,cols))
    #fill matrix with values

    for l in range(len(lines)):
        tmp_list = re.findall("\d+\.\d+", lines[l])
        tmp_f_list = [float(i) for i in tmp_list]
        #print()
        for c in range(len(tmp_list)):
            #print(matrix[l][c])
            #print(tmp_f_list[c])
            if not tmp_f_list[c] == 0.0:
                matrix[l][c] = tmp_f_list[c]

    #print(matrix)
    #print(len(image_list))
    #print("parse_datafile dtype: "+str(matrix.dtype))
    return matrix




def main():
    print("Starting calculation of distance transformations...")
    #get all matrix files
    all_files = list_data(PATH_DATA)
    #parse data
    parsed_data = []
    for f in range(len(all_files)):
        tmp_data = parse_datafile(all_files[f])
        #np.where(tmp_data < 0.5, tmp_data, tmp_data*0)#remove noise, not needed
        tmp_data = ((tmp_data < 0.5)*0.0)+((tmp_data > 0.5))
        parsed_data.append((tmp_data*255).astype(np.uint8))# to make a grayimage out of it
        #if f == 0:
            #print(tmp_data)

    #calculate distance images
    dist1_list = []
    dist2_list = []

    for p in range(len(parsed_data)):
        input_dis = np.expand_dims(parsed_data[p],axis=-1)
        if p == 0:
            #print(parsed_data[p].shape)
            print(input_dis.shape)
            #print(input_dis)
        dist1_tmp = cv.distanceTransform(cv.bitwise_not(input_dis), distanceType=cv.DIST_L1,maskSize=3)
        dist2_tmp = cv.distanceTransform(cv.bitwise_not(input_dis), distanceType=cv.DIST_L2,maskSize=5)
        norm_image1 = cv.normalize(dist1_tmp, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
        norm_image2 = cv.normalize(dist2_tmp, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
        dist1_list.append(norm_image1)
        dist2_list.append(norm_image2)
        #DEBUG
        if p == 0:
            pass
            #print(dist1_tmp)
            #print(norm_image1)
            #print(norm_image2)
            '''
            cv.namedWindow('input',cv.WINDOW_NORMAL)
            cv.namedWindow('dist1',cv.WINDOW_NORMAL)
            cv.namedWindow('dist2',cv.WINDOW_NORMAL)
            cv.imshow('input', input_dis)
            cv.imshow('dist1', norm_image1)
            cv.imshow('dist2', norm_image2)
            cv.waitKey(0)
            cv.destroyAllWindows()
            '''
    #write distance images
    print("Writing dist1:"+str(len(dist1_list))+" dist2:"+str(len(dist2_list)))

    for i in range(len(dist1_list)):
        write_dist(i,1,dist1_list[i],PATH_DISTANCES)
        write_dist(i,2,dist2_list[i],PATH_DISTANCES)


if __name__ == '__main__':
    main()
