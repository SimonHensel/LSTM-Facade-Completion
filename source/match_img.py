import numpy as np
import os
import re

TEST_TXT_PATH = "/home/simon/Arbeit/mdlstm_modified/tensorflow-multi-dimensional-lstm/results/eval_graz_25/y_000000.txt"

#ORIGIN_PATH = "/home/simon/Arbeit/mdlstm_modified/tensorflow-multi-dimensional-lstm/data/graz50/graz50_matrix/" #HOME
ORIGIN_PATH = "/media/DATA/simon/Repos/facadecompletion/data/graz50/graz50_matrix/"

ZERO_PADDING_SIZE = 25

def list_data(input_path, text):
    image_list = []
    file_list = os.listdir(input_path)

    for i in range(len(file_list)):
        current_file_abs_path = input_path+file_list[i]
        if (current_file_abs_path.endswith(text)):
            image_list.append(current_file_abs_path)

    image_list.sort()

    return image_list

def parse_datafile(file_to_read):
    #print(file_to_read)

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
    matrix = np.zeros((rows,cols),dtype=float)
    #matrix = np.random.uniform(low=0.001, high=0.1, size=(rows,cols))
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

def zero_pad_in(matrix):
    #print(matrix.shape)
    matrix = np.array(matrix)
    #print(matrix.shape)
    new_matrix = np.zeros((ZERO_PADDING_SIZE,ZERO_PADDING_SIZE))
    #print(matrix)

    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            new_matrix[y][x] = matrix[y][x]

    return new_matrix

def compare_matrix(m1,m2):
    if not np.all(m1.shape == m2.shape):
        print("Shape "+str(m1.shape)+" and "+str(m2.shape)+" do not match!")
        return False
    else:
        print("Same Shape")
        print(m1)
        print("##########")
        print(m2)
        y_len = m1.shape[0]
        x_len = m2.shape[1]

        for y in range(y_len):
            for x in range(x_len):
                if not m1[y][x] == m2[y][x]:
                    print("Missmatch at "+str(y)+" "+str(x))
                    return False
    print("Match found!")
    return True




def match_img(matrix_path):
    txt_list = list_data(ORIGIN_PATH,".txt")
    spatial_list = list_data(ORIGIN_PATH,"_spatial.txt")
    txt_list = [item for item in txt_list if item not in spatial_list]#txt_list-spatial_list
    img_list = list_data(ORIGIN_PATH,"facade.png")

    #print(txt_list)

    to_match = parse_datafile(matrix_path)

    to_match = (to_match > 0.5)*to_match

    #print(to_match)
    #print("##########")


    found_number = -1

    for i in range(len(txt_list)):
        tmp = parse_datafile(txt_list[i])
        if tmp.shape[0] <= ZERO_PADDING_SIZE and tmp.shape[1] <= ZERO_PADDING_SIZE:
            tmp = zero_pad_in(tmp)

            #print(tmp)
            #exit()
            #tmp_swaped = np.swapaxes(tmp,0,1)

            if np.array_equal(to_match,tmp):
                found_number = i

    return found_number

def match_matrix(to_match):
    #print(ORIGIN_PATH)
    txt_list = list_data(ORIGIN_PATH,".txt")
    spatial_list = list_data(ORIGIN_PATH,"_spatial.txt")
    txt_list = [item for item in txt_list if item not in spatial_list]#txt_list-spatial_list
    img_list = list_data(ORIGIN_PATH,"facade.png")

    to_match = (to_match > 0.5)*to_match

    #print(to_match)
    #print("##########")

    found_number = -1
    #print(txt_list)

    for i in range(len(txt_list)):
        tmp = parse_datafile(txt_list[i])
        if tmp.shape[0] <= ZERO_PADDING_SIZE and tmp.shape[1] <= ZERO_PADDING_SIZE:
            tmp = zero_pad_in(tmp)

            #print(tmp)
            #exit()
            #tmp_swaped = np.swapaxes(tmp,0,1)

            if np.array_equal(to_match,tmp):
                found_number = i

    return found_number

def compare_files(path1,path2):
    f1=open(path1,"r")
    f2=open(path2,"r")
    for line1 in f1:
        counter = 0
        print("######################")
        for line2 in f2:
            if line1==line2:
                print("SAME\n")
                counter = counter+1
            else:
                print(line1 + line2)
                counter = -1
                break
        if counter > 0:
            return True

    f1.close()
    f2.close()
    return False


def match_file(matrix_path):
    txt_list = list_data(ORIGIN_PATH,".txt")
    img_list = list_data(ORIGIN_PATH,"facade.png")

    #to_match = parse_datafile(matrix_path)

    to_match = (to_match > 0.5)*to_match

    #print(to_match)

    found_number = -1

    for i in range(len(txt_list)):
        #tmp = parse_datafile(txt_list[i])
        if compare_files(matrix_path,txt_list[i]):
            found_number = i

    return found_number


if __name__ == "__main__":
    tmp = match_img(TEST_TXT_PATH)
    #tmp = match_file(TEST_TXT_PATH)

    print(tmp)
