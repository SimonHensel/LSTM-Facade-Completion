import numpy as np
import os
import re
#from scipy import interpolate
import cv2 as cv



"""

IDEE: cluster objetcs and apply mdlstm only to the clusters.

____________
|          |
| D  D  D  |
| D     D  |
|    D  D  |
|__________|


Facade with missing windows


____________
|          |
| D  D  D  |
| D  D  D  |
| D  D  D  |
|__________|

Facade with full windows
"""

'''
def random_short_diagonal_matrix(h, w):
    m = np.random.uniform(low=0.0, high=0.1, size=(h, w))

    x1_x = np.random.randint(low=1, high=w - 1)
    x1_y = np.random.randint(low=1, high=h - 1)

    x2_x = x1_x + 1
    x2_y = x1_y + 1

    m[x1_x, x1_y] = 1.0
    m[x2_x, x2_y] = 1.0

    return m
'''

PATH_DATA = "/media/DATA/simon/mdlstm_modified/tensorflow-multi-dimensional-lstm/data/tests/"
#PATH_DATA = "/home/simon/Dropbox/Arbeit/P6000/mdlstm_modified/tensorflow-multi-dimensional-lstm/data/tests/"
PATH_OUTPUT = "/media/DATA/simon/mdlstm_modified/tensorflow-multi-dimensional-lstm/out/"

ZERO_PADDING_SIZE = 32#128#512
MAX_DIFICULTY = 8
MIN_DIFICULTY = 1
MAX_EDGE = 4
MIN_EDGE = 1
MAX_COLS = 12
MIN_COLS = 6
MAX_ROWS = 12
MIN_ROWS = 6



def zero_pad_in(matrix):
    new_matrix = np.random.uniform(low=0.001, high=0.1, size=(ZERO_PADDING_SIZE,ZERO_PADDING_SIZE))

    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            new_matrix[y][x] = matrix[y][x]

    return new_matrix

def scale_up(matrix): #maybe better option, who knows
    cols = matrix.shape[0]
    rows = matrix.shape[1]

    resized_m = cv.resize(matrix,(ZERO_PADDING_SIZE,ZERO_PADDING_SIZE),interpolation=cv.INTER_AREA)

    return resized_m


def generate_random_facade():
    edge = np.random.randint(low=MIN_EDGE, high=MAX_EDGE)
    rows = np.random.randint(low=MIN_ROWS, high=MAX_ROWS)
    cols = np.random.randint(low=MIN_COLS, high=MAX_COLS)

    #matrix = np.zeros((edge*2+rows*2,edge*2+cols*2), dtype=np.float32)
    matrix = np.random.uniform(low=0.001, high=0.1, size=(edge*2+rows*2,edge*2+cols*2))

    for y in range(1,rows):
        for x in range(1,cols):
            matrix[edge+y*2][edge+x*2] = 1.0

    return matrix

def random_erase(matrix_list):
    new_matrix_list = []
    for x in range(len(matrix_list)):
        number_of_objects = np.random.randint(low=MIN_DIFICULTY, high=MAX_DIFICULTY)
        matrix = matrix_list[x]
        #print(matrix.shape)

        y_len = matrix.shape[0]
        x_len = matrix.shape[1]

        new_matrix = matrix.copy()

        for i in range(number_of_objects):
            #get random window
            selected_value = 0
            while not selected_value == 1.0: #TODO make more efficient
                x_rand = np.random.randint(low=0, high=x_len)
                y_rand = np.random.randint(low=0, high=y_len)

                selected_value = matrix[y_rand][x_rand]

            #print("Random Object is "+str(x_rand)+","+str(y_rand))

            neighbors = []
            neighbors.append([y_rand,x_rand])

            while neighbors :
                #print(neighbors[len(neighbors)-1])
                y,x = neighbors[len(neighbors)-1]
                neighbors.pop()
                #print(y)
                #print(x)
                #ONLY DIRECT NEIGHBORS
                if  y < y_len-1 and new_matrix[y+1][x] == 1.0:
                    neighbors.append([y+1,x])
                if x < x_len-1 and new_matrix[y][x+1] == 1.0:
                    neighbors.append([y,x+1])
                if y > 0 and new_matrix[y-1][x] == 1.0:
                    neighbors.append([y-1,x])
                if x > 0 and new_matrix[y][x-1] == 1.0:
                    neighbors.append([y,x-1])

                new_matrix[y][x] = np.random.uniform(0.001,0.1) #set current position to zero

        new_matrix_list.append(new_matrix)

    return new_matrix_list

def next_batch(bs):
    x = []
    y = []
    y_tmp = []
    for i in range(bs):
        y_tmp.append(generate_random_facade())

    x_tmp = np.array(random_erase(y_tmp))
    y_tmp = np.array(y_tmp)


    for i in range(bs):
        x.append(zero_pad_in(x_tmp[i]))
        y.append(zero_pad_in(y_tmp[i]))
        #x.append(scale_up(x_tmp[i]))
        #y.append(scale_up(y_tmp[i]))
    #y = np.roll(x, shift=-1, axis=2)

    t = get_relevant_prediction_index(y)
    x = np.array(x)
    y = np.array(y)
    if np.any(np.isin(x,0.0)) or np.any(np.isin(y,0.0)):
        print("ERROR: Zero found!")
        exit()
    return x, y, t


def visualise_mat(m):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(m.real, cmap='jet', interpolation='none')
    plt.show()

def write_mat(m, file_name):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(m.real, cmap='jet', interpolation='none')
    plt.savefig(PATH_OUTPUT+file_name)
    #plt.show()

def visualise_compare(m1,m2):
    import matplotlib.pyplot as plt
    columns = 2
    rows = 1
    fig = plt.figure()
    fig.add_subplot(rows, columns, 1)
    plt.imshow(m1.real, cmap='jet', interpolation='none')
    fig.add_subplot(rows, columns, 2)
    plt.imshow(m2.real, cmap='jet', interpolation='none')
    plt.show()


def find_target_for_matrix(y_):
    w_y = 1#np.where(y_ == 1)[1][1]
    h_y = 1#np.where(y_ == 1)[0][1]
    return w_y, h_y


def get_relevant_prediction_index(y_):
    a = []
    for yy_ in y_:
        a.append(find_target_for_matrix(yy_))
    return np.array(a)


if __name__ == '__main__':
    x_, y_, t_ = next_batch(bs=1)
    #visualise_mat(x_[0])
    #visualise_mat(y_[0])
    visualise_compare(x_[0],y_[0])
