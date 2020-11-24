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

PATH_DATA = "/media/DATA/simon/mdlstm_modified/tensorflow-multi-dimensional-lstm/data/tests_cluster/"
PATH_DISTANCES = "/media/DATA/simon/mdlstm_modified/tensorflow-multi-dimensional-lstm/data/Distances/"
PATH_COMPRESSION_INFO = "/media/DATA/simon/mdlstm_modified/tensorflow-multi-dimensional-lstm/data/compress_information/"
#PATH_DATA = "/home/simon/Dropbox/Arbeit/P6000/mdlstm_modified/tensorflow-multi-dimensional-lstm/data/tests/"
PATH_OUTPUT = "/media/DATA/simon/mdlstm_modified/tensorflow-multi-dimensional-lstm/out_grid25/"
#PATH_OUTPUT = "/media/DATA/simon/mdlstm_modified/tensorflow-multi-dimensional-lstm/out_md/"
EVAL_PATH = "/media/DATA/simon/mdlstm_modified/tensorflow-multi-dimensional-lstm/eval_25/"
'''
PATH_DATA = "/home/simon/tests_cluster"
PATH_OUTPUT = "/home/simon/output"
'''

EVAL_SIZE = 10000
ZERO_PADDING_SIZE = 25#64#512
DIFICULTY = 8
DIFICULTY_PERCENTAGE = 0.3
DIFICULTY_MAX = 12
DIFICULTY_MIN = 8
GHOST_MIN = 0.4
GHOST_MAX = 0.6
MAX_FALSE = 3
MAX_FALSE_SIZE = 3
MAX_MISSING_COORDINATES = 32

def list_data(input_path, begins_with=None):
        #print("PATH %s" % input_path)
    #input_path = input_path+"/"
    if begins_with == None:
        begins_with='0'
    image_list = []
    file_list = os.listdir(input_path) # Change this PATH to traverse other directories if you want.
    if file_list != None:
        pass
    #print("%s files were found under current folder. " % len(file_list))
    #print("Please be noted that only files end with '*.jpg' will be load!")
    for i in range(len(file_list)):
        current_file_abs_path = input_path+file_list[i]
        if (current_file_abs_path.endswith(".txt") and (not current_file_abs_path.startswith('c'))):
            if begins_with == None:
                image_list.append(current_file_abs_path)
                #print("Found %s successfully!" % current_file_abs_path)
            else:
                if (file_list[i].startswith(begins_with)):
                    image_list.append(current_file_abs_path)
    #print("Cannot find any JPEG files, please check the path.")
    image_list.sort()
    #print("#####################################")
    #print(len(image_list))
    #print(len(file_list))
    #print("#####################################")
    return image_list

def list_dist(input_path, number):
        #print("PATH %s" % input_path)
    #input_path = input_path+"/"
    image_list = []
    file_list = os.listdir(input_path) # Change this PATH to traverse other directories if you want.
    if file_list != None:
        pass
    #print("%s files were found under current folder. " % len(file_list))
    #print("Please be noted that only files end with '*.jpg' will be load!")
    for i in range(len(file_list)):
        current_file_abs_path = input_path+file_list[i]
        #print(file_list[i])
        if (file_list[i].endswith(".txt") and file_list[i].startswith(str(number))):
            #print(file_list[i])
            image_list.append(current_file_abs_path)

    #print("Cannot find any JPEG files, please check the path.")
    image_list.sort()
    #print("#####################################")
    #print(image_list[0])
    #print(len(image_list))
    #print(len(file_list))
    #print("#####################################")
    return image_list

def list_coordinate(input_path, begins_with=None):
        #print("PATH %s" % input_path)
    #input_path = input_path+"/"
    image_list = []
    begins_with = 'c'
    file_list = os.listdir(input_path) # Change this PATH to traverse other directories if you want.
    if file_list != None:
        pass
    #print("%s files were found under current folder. " % len(file_list))
    #print("Please be noted that only files end with '*.jpg' will be load!")
    for i in range(len(file_list)):
        current_file_abs_path = input_path+file_list[i]
        if (current_file_abs_path.endswith(".txt")):
            if begins_with == None:
                image_list.append(current_file_abs_path)
                #print("Found %s successfully!" % current_file_abs_path)
            else:
                if (file_list[i].startswith(begins_with)):
                    image_list.append(current_file_abs_path)
        else:
            pass
    if len(image_list) != 0:
        for list_index in range(len(image_list)):
            pass
        #print(image_list[list_index])
    else:
        pass
    #print("Cannot find any JPEG files, please check the path.")
    image_list.sort()
    #print("#####################################")
    #print(len(image_list))
    #print("#####################################")
    return image_list

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

def zero_pad_in(matrix):
    matrix = np.array(matrix)
    #print(matrix.shape)
    new_matrix = np.random.uniform(low=0.001, high=0.1, size=(ZERO_PADDING_SIZE,ZERO_PADDING_SIZE))
    #print(matrix)

    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            new_matrix[y][x] = matrix[y][x]

    return new_matrix


def read_random_matrix():
    data_list = list_data(PATH_DATA)
    coordinate_list = list_coordinate(PATH_DATA)


    rand_number = np.random.randint(low=0, high=len(coordinate_list)-1)
    matrix = parse_datafile(data_list[rand_number])
    coordinates_array = read_coordinates(rand_number)

    while max(matrix.shape[0],matrix.shape[1]) > ZERO_PADDING_SIZE:
        #print("Too Big!")
        rand_number = None
        matrix = None
        rand_number = np.random.randint(low=0, high=len(data_list)-1)
        matrix = parse_datafile(data_list[rand_number])
        coordinates_array = read_coordinates(rand_number)
    #print(matrix.shape)
    #print("read_random_matrix dtype: "+str(matrix.dtype))

    return matrix, coordinates_array, rand_number

def read_random_matrix_with_distance():
    data_list = list_data(PATH_DATA)
    coordinate_list = list_coordinate(PATH_DATA)
    dist1_list = list_dist(PATH_DISTANCES,1)
    dist2_list = list_dist(PATH_DISTANCES,2)

    rand_number = np.random.randint(low=0, high=len(coordinate_list)-1)
    #print(str(data_list[rand_number])+"\n"+str(dist1_list[rand_number])+"\n"+str(dist2_list[rand_number]))
    matrix = parse_datafile(data_list[rand_number])
    dist1 = parse_datafile(dist1_list[rand_number])
    dist2 = parse_datafile(dist2_list[rand_number])
    #print(str(matrix.shape)+"\n"+str(dist1.shape)+"\n"+str(dist2.shape))
    coordinates_array = read_coordinates(rand_number)

    while max(matrix.shape[0],matrix.shape[1]) > ZERO_PADDING_SIZE:
        #print("Too Big!")
        rand_number = None
        matrix = None
        rand_number = np.random.randint(low=0, high=len(data_list)-1)
        matrix = parse_datafile(data_list[rand_number])
        dist1 = parse_datafile(dist1_list[rand_number])
        dist2 = parse_datafile(dist2_list[rand_number])
        #coordinates_array = read_coordinates(rand_number)
    #print(matrix.shape)
    #print("read_random_matrix dtype: "+str(matrix.dtype))

    return matrix, dist1, dist2 #,coordinates_array



def count_objects(matrix):
    #img = cv2.imread('ba3g0.jpg')
    gray = np.array(matrix*255,dtype=np.uint8)
    ret, thresh = cv.threshold(gray, 127, 255, 0)
    #im2, contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    #for cnt in contours:
      #cv2.drawContours(img,[cnt],0,(0,0,255),1)
    #print(len(contours))
    return len(contours)

def make_predonly_y(matrix_list, perc):
    new_matrix_list = []
    y_matrix_list = []
    for x in range(len(matrix_list)):
        matrix = matrix_list[x]
        number_of_objects = int(count_objects(matrix)*perc)
        #print(matrix.shape)

        y_len = matrix.shape[0]
        x_len = matrix.shape[1]

        new_matrix = matrix.copy()
        y_matrix = np.random.uniform(0.001,0.1,size=matrix.shape)
        #x_rand = 0
        #y_rand = 0

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
                y_matrix[y][x] = 1.0

        new_matrix_list.append(new_matrix)
        y_matrix_list.append(y_matrix)
        #print(len(new_matrix_list))
        #print(len(y_matrix_list))

    return new_matrix_list, y_matrix_list


def random_erase_percentage(matrix_list, perc, coordinate_array=None):
    new_matrix_list = []
    erased_objects = []
    for x in range(len(matrix_list)):
        matrix = matrix_list[x]
        number_of_objects = int(count_objects(matrix)*perc)
        if number_of_objects < 1:
            number_of_objects = 1

        if number_of_objects > (MAX_MISSING_COORDINATES/2):
            number_of_objects = int(MAX_MISSING_COORDINATES/2)
            #print(number_of_objects)
        #print("matrix_shape: "+str(matrix.shape))

        y_len = matrix.shape[0]
        x_len = matrix.shape[1]

        new_matrix = matrix.copy()
        #x_rand = 0
        #y_rand = 0

        for i in range(number_of_objects):
            erased_objects_part = []
            #get random window
            selected_value = 0
            while not selected_value == 1.0: #TODO make more efficient
            #    if x_rand < x_len-1:
            #        x_rand += 1
            #    else:
            #        x_rand = 0
            #        y_rand += 1
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
                if coordinate_array:
                    erased_objects_part.append([float(x),float(y)])

        while len(erased_objects_part) < MAX_MISSING_COORDINATES and coordinate_array: #FILL WITH ZEROS
            #print(len(erased_objects_part))
            erased_objects_part.append([np.random.uniform(0.001,0.1),np.random.uniform(0.001,0.1)])

        if len(erased_objects_part) > MAX_MISSING_COORDINATES and coordinate_array:
            print("ERROR len(erased_objects_part) > MAX_MISSING_COORDINATES")
            print(len(erased_objects_part))
            exit()

        if coordinate_array:
            erased_objects.append(erased_objects_part)
        new_matrix_list.append(new_matrix)

    if coordinate_array:
        for o in range(len(erased_objects)):
            erased_objects[o]= coordinate_sort(erased_objects[o])
            erased_objects[o] = np.array(erased_objects[o])
            #print("erased_objects[o] shape: "+str(erased_objects[o].shape))
            #erased_objects[o].flatten()
            #print("erased_objects[o] shape: "+str(erased_objects[o].shape))

        erased_objects = np.array(erased_objects)
        erased_objects = erased_objects.reshape((erased_objects.shape[0],erased_objects.shape[1]*erased_objects.shape[2]))
        #print("erased_objects FINAL SHAPE: "+str(erased_objects.shape))

    return new_matrix_list, erased_objects

def random_ghosting_erase(matrix_list, perc, coordinate_array=None):
    new_matrix_list = []
    erased_objects = []
    for x in range(len(matrix_list)):
        matrix = matrix_list[x]
        number_of_objects = int(count_objects(matrix)*perc)
        #print("Erased Objects: "+str(number_of_objects))
        if number_of_objects < 1:
            number_of_objects = 1

        if number_of_objects > (MAX_MISSING_COORDINATES/2):
            number_of_objects = int(MAX_MISSING_COORDINATES/2)
            #print(number_of_objects)
        #print(matrix.shape)

        y_len = matrix.shape[0]
        x_len = matrix.shape[1]

        new_matrix = matrix.copy()
        #x_rand = 0
        #y_rand = 0

        for i in range(number_of_objects):
            erased_objects_part = []
            ghost_value = np.random.uniform(0.0 ,GHOST_MAX)
            #print("ghost_value: "+str(ghost_value))
            #get random window
            selected_value = 0
            while not selected_value == 1.0: #TODO make more efficient
            #    if x_rand < x_len-1:
            #        x_rand += 1
            #    else:
            #        x_rand = 0
            #        y_rand += 1
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

                if ghost_value <= GHOST_MIN: #ANSTATT GANZ ZU  WERT VERRINGERN
                    new_matrix[y][x] = np.random.uniform(0.001,0.1) #set current position to zero
                else:
                    new_matrix[y][x] = ghost_value
                if coordinate_array:
                    erased_objects_part.append([float(x),float(y)])

        while len(erased_objects_part) < MAX_MISSING_COORDINATES and coordinate_array: #FILL WITH ZEROS
            #print(len(erased_objects_part))
            erased_objects_part.append([np.random.uniform(0.001,0.1),np.random.uniform(0.001,0.1)])

        if len(erased_objects_part) > MAX_MISSING_COORDINATES and coordinate_array:
            print("ERROR len(erased_objects_part) > MAX_MISSING_COORDINATES")
            print(len(erased_objects_part))
            exit()

        ###ADDING GHOSTS############################################################
        for f in range(MAX_FALSE):
            if (new_matrix.shape[0]-1) > (MAX_FALSE_SIZE+1) and (new_matrix.shape[1]-1) > (MAX_FALSE_SIZE+1):
                x_max = np.random.randint(low=MAX_FALSE_SIZE+1, high=new_matrix.shape[1]-1)
                y_max = np.random.randint(low=MAX_FALSE_SIZE+1, high=new_matrix.shape[0]-1)
                x_min = np.random.randint(low=x_max-MAX_FALSE_SIZE, high=x_max-1)
                y_min = np.random.randint(low=y_max-MAX_FALSE_SIZE, high=y_max-1)
                #print("+++++++++")
                #print(x_max)
                #print(y_max)
                #print(x_min)
                #print(y_min)
                #print("---------")
                tmp_ghost_value = np.random.uniform(GHOST_MIN,GHOST_MAX)
                #print(tmp_ghost_value)
                #print(new_matrix[y_min:y_max,x_min:x_max])
                new_matrix[y_min:y_max,x_min:x_max] = tmp_ghost_value
                #print("---------")
                #print(new_matrix)

        erased_objects.append(erased_objects_part)
        new_matrix_list.append(new_matrix)

    if coordinate_array:
        for o in range(len(erased_objects)):
            erased_objects[o]= coordinate_sort(erased_objects[o])
            erased_objects[o] = np.array(erased_objects[o])
            #print("erased_objects[o] shape: "+str(erased_objects[o].shape))
            #erased_objects[o].flatten()
            #print("erased_objects[o] shape: "+str(erased_objects[o].shape))

        erased_objects = np.array(erased_objects)
        erased_objects = erased_objects.reshape((erased_objects.shape[0],erased_objects.shape[1]*erased_objects.shape[2]))
        #print("erased_objects FINAL SHAPE: "+str(erased_objects.shape))



    return new_matrix_list, erased_objects

def to_dez(coordinate):
    result = 0
    if len(coordinate) == 2:
        result = coordinate[0]*10+coordinate[1]
    else:
        print("ERROR: to_dez")
        print(len(coordinate))
        exit()
    return result

def coordinate_sort(coordinate_list):
    sorted_list = []

    while coordinate_list:
        min = 999
        pos = 0
        for i in range(len(coordinate_list)):
            dez = to_dez(coordinate_list[i])
            if dez < min:
                min = to_dez(coordinate_list[i])
                pos = i
        sorted_list.append(coordinate_list[i])
        coordinate_list.pop(i)

    return sorted_list

def read_coordinates(iter):
    path = PATH_DATA+"c{0:06d}".format(iter)+".txt"

    coordinate_array = []

    lines = [line.rstrip('\n') for line in open(path)]

    #get sizes
    #print(file_to_read)
    rows = len(lines)
    cols = 2

    for l in range(len(lines)):
        tmp_list = re.findall("\d+\.\d+", lines[l])
        for c in range(len(tmp_list)):
            coordinate_array.append(tmp_list[c])

    return np.array(coordinate_array)




def next_batch_c(bs):
    x = []
    y = []
    y_tmp = []
    x_tmp = []
    for i in range(bs):
        random_matrix, random_coordinates, _ = read_random_matrix()
        y_tmp.append(random_coordinates)
        x_tmp.append(zero_pad_in(random_matrix))

    #x_tmp = np.array(random_erase(y_tmp,np.random.randint(DIFICULTY_MIN,DIFICULTY_MAX)))
    x_matrix, erased_objects = random_erase_percentage(x_tmp,DIFICULTY_PERCENTAGE)
    x_tmp = np.array(x_matrix)
    #x_tmp, y_tmp = make_predonly_y(y_tmp,DIFICULTY_PERCENTAGE)
    #print(len(x_tmp))
    #print(len(y_tmp))

    y = np.array(erased_objects)
    x = np.array(x_tmp)

    #print(y.shape)
    #print(x.shape)


    #for i in range(bs):
        #print(i)
        #x.append(zero_pad_in(x_tmp[i]))
        #y.append(zero_pad_in(y_tmp[i]))
        #x.append(interpolate(x_tmp[i], ZERO_PADDING_SIZE))
        #y.append(interpolate(y_tmp[i], ZERO_PADDING_SIZE))
    #y = np.roll(x, shift=-1, axis=2)

    t = get_relevant_prediction_index(y)
    #x = np.array(x)
    #y = np.array(y)
    if np.any(np.isin(x,0.0)) or np.any(np.isin(y,0.0)):
        print("ERROR: Zero found!")
        exit()

    #print("x dtype: "+str(x.dtype))
    #print("y dtype: "+str(y.dtype))
    return x, y, t

def next_batch_d(bs):
    x = []
    y = []
    y_dist1 = []
    y_dist2 = []
    dist1_array = []
    dist2_array = []
    y_tmp = []
    for i in range(bs):
        random_matrix, dist1, dist2 = read_random_matrix_with_distance()                                 #random matrix + distance matrizen
        #print("random_matrix dtype: "+str(random_matrix.dtype))
        y_tmp.append(random_matrix)
        dist1_array.append(dist1)
        dist2_array.append(dist2)
        #x_tmp.append(random_matrix)

    #x_tmp = np.array(random_erase(y_tmp,np.random.randint(DIFICULTY_MIN,DIFICULTY_MAX)))
    x_matrix, _ = random_erase_percentage(y_tmp,DIFICULTY_PERCENTAGE)
    #x_matrix, _ = random_ghosting_erase(y_tmp,DIFICULTY_PERCENTAGE)
    #print("x_matrix shape: "+str(len(x_matrix))+" "+str(len(x_matrix[0]))+" "+str(len(x_matrix[0])))
    x_array = np.array(x_matrix)
    dist1_array = np.array(dist1_array)
    dist2_array = np.array(dist2_array)
    #x_tmp, y_tmp = make_predonly_y(y_tmp,DIFICULTY_PERCENTAGE)
    #print(len(x_tmp))
    #print(len(y_tmp))

    y_array = np.array(y_tmp)
    #x_array = np.asarray(x_tmp)
    '''
    print("before zero pad")
    print(x_array.shape)
    print(y_array.shape)
    print(dist1_array.shape)
    print(dist2_array.shape)
    '''

    for i in range(bs):
        #print(i)
        x.append(zero_pad_in(x_array[i]))
        y_dist1.append(zero_pad_in(dist1_array[i]))
        y_dist2.append(zero_pad_in(dist2_array[i]))
        y.append(zero_pad_in(y_array[i]))
        #print("x_array[i] dtype: "+str(x_array[i].dtype))
        #print("y_array[i] dtype: "+str(y_array[i].dtype))
        #x.append(interpolate(x_array[i], ZERO_PADDING_SIZE))
        #y.append(interpolate(y_array[i], ZERO_PADDING_SIZE))
    #y = np.roll(x, shift=-1, axis=2)

    t = get_relevant_prediction_index(y)
    x = np.array(x)
    y = np.array(y)
    y_dist1 = np.array(y_dist1)
    y_dist2 = np.array(y_dist2)
    if np.any(np.isin(x,0.0)) or np.any(np.isin(y,0.0)):
        print("ERROR: Zero found!")
        exit()
    #print("x dtype: "+str(x.dtype))
    #print("y dtype: "+str(y.dtype))
    '''
    print("return batch")
    print(x.shape)
    print(y.shape)
    print(y_dist1.shape)
    print(y_dist2.shape)
    '''

    return x, y, y_dist1, y_dist2, t

def next_batch(bs):
    x = []
    y = []
    y_tmp = []
    for i in range(bs):
        random_matrix, _ , _ = read_random_matrix()
        #print("random_matrix dtype: "+str(random_matrix.dtype))
        y_tmp.append(random_matrix)
        #x_tmp.append(random_matrix)

    #x_tmp = np.array(random_erase(y_tmp,np.random.randint(DIFICULTY_MIN,DIFICULTY_MAX)))
    x_matrix, _ = random_erase_percentage(y_tmp,DIFICULTY_PERCENTAGE)
    #x_matrix, _ = random_ghosting_erase(y_tmp,DIFICULTY_PERCENTAGE)
    #print("x_matrix shape: "+str(len(x_matrix))+" "+str(len(x_matrix[0]))+" "+str(len(x_matrix[0])))
    x_array = np.asarray(x_matrix)
    #x_tmp, y_tmp = make_predonly_y(y_tmp,DIFICULTY_PERCENTAGE)
    #print(len(x_tmp))
    #print(len(y_tmp))

    y_array = np.asarray(y_tmp)
    #x_array = np.asarray(x_tmp)

    for i in range(bs):
        #print(i)
        x.append(zero_pad_in(x_array[i]))
        y.append(zero_pad_in(y_array[i]))
        #print("x_array[i] dtype: "+str(x_array[i].dtype))
        #print("y_array[i] dtype: "+str(y_array[i].dtype))
        #x.append(interpolate(x_array[i], ZERO_PADDING_SIZE))
        #y.append(interpolate(y_array[i], ZERO_PADDING_SIZE))
    #y = np.roll(x, shift=-1, axis=2)

    t = get_relevant_prediction_index(y)
    x = np.array(x)
    y = np.array(y)
    if np.any(np.isin(x,0.0)) or np.any(np.isin(y,0.0)):
        print("ERROR: Zero found!")
        exit()
    #print("x dtype: "+str(x.dtype))
    #print("y dtype: "+str(y.dtype))

    return x, y, t

def compress(batches, numbers):
    """
    Compresses the Matrices according to the compression files
    """
    #print("compressing...")
    #print(batches[0])
    batches = np.array(batches)
    new_batch = []
    #print("Batches: "+str(batches.shape[0]))
    #print("Files: "+str(len(numbers)))
    for i in range(len(numbers)):
        infile = PATH_COMPRESSION_INFO+"{0:06d}".format(numbers[i])+".txt"

        # READ FILESw
        information = open(infile)
        lines = [line.rstrip('\n') for line in information]
        rows = len(lines)
        if not rows == 2:
            print("Error: to much lines - "+ str(rows))
        #print(lines[0])
        #print(lines[1])
        y_len = len(re.findall("\d+", lines[0]))
        x_len = len(re.findall("\d+", lines[1]))

        y_info = list(map(int,re.findall("\d+", lines[0])))
        x_info = list(map(int,re.findall("\d+", lines[1])))
        information.close()
        #print(y_info)
        #print(y_len)
        #print(x_info)
        #print(x_len)
        #print(batches.shape)

        #DO THE COMPRESSSION
        new_matrix = np.copy(batches[i])
        counter = 0
        for y in range(y_len):
            new_matrix = np.delete(new_matrix,y_info[y]-counter,0)
            counter += 1

        counter = 0
        for x in range(x_len):
            new_matrix = np.delete(new_matrix,x_info[x]-counter,1)
            counter += 1
        new_batch.append(new_matrix)

    #print(new_matrix[0])
    #visualise_compare(x_[0],y_[0])
    #print(new_batch.shape)

    return np.array(new_batch)

def next_comp_batch(bs):
    #print("next_comp_batch")
    x = []
    y = []
    y_tmp = []
    numbers = []
    for i in range(bs):
        random_matrix, _, rand_number= read_random_matrix()
        #print("random_matrix dtype: "+str(random_matrix.dtype))
        y_tmp.append(random_matrix)
        numbers.append(rand_number)
        #x_tmp.append(random_matrix)

    #x_tmp = np.array(random_erase(y_tmp,np.random.randint(DIFICULTY_MIN,DIFICULTY_MAX)))

    x_matrix, _ = random_erase_percentage(y_tmp,DIFICULTY_PERCENTAGE)
    y_tmp = compress(y_tmp, numbers)
    x_matrix = compress(x_matrix, numbers)
    #x_matrix, _ = random_ghosting_erase(y_tmp,DIFICULTY_PERCENTAGE)
    #print("x_matrix shape: "+str(len(x_matrix))+" "+str(len(x_matrix[0]))+" "+str(len(x_matrix[0])))
    x_array = np.asarray(x_matrix)
    #x_tmp, y_tmp = make_predonly_y(y_tmp,DIFICULTY_PERCENTAGE)
    #print(len(x_tmp))
    #print(len(y_tmp))

    y_array = np.asarray(y_tmp)
    #x_array = np.asarray(x_tmp)

    for i in range(bs):
        #print(i)
        x.append(zero_pad_in(x_array[i]))
        y.append(zero_pad_in(y_array[i]))
        #print("x_array[i] dtype: "+str(x_array[i].dtype))
        #print("y_array[i] dtype: "+str(y_array[i].dtype))
        #x.append(interpolate(x_array[i], ZERO_PADDING_SIZE))
        #y.append(interpolate(y_array[i], ZERO_PADDING_SIZE))
    #y = np.roll(x, shift=-1, axis=2)

    t = get_relevant_prediction_index(y)
    x = np.array(x)
    y = np.array(y)
    if np.any(np.isin(x,0.0)) or np.any(np.isin(y,0.0)):
        print("ERROR: Zero found!")
        exit()
    #print("x dtype: "+str(x.dtype))
    #print("y dtype: "+str(y.dtype))

    return x, y, t


def visualise_mat(m):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(m.real, cmap='jet', interpolation='none')
    plt.show()
    plt.close()

def write_mat(m, file_name):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(m.real, cmap='jet', interpolation='none')
    plt.savefig(PATH_OUTPUT+file_name)
    plt.close()
    #plt.show()

def write_mat_conf(m, file_name, path):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(m.real, cmap='jet', interpolation='none')
    plt.savefig(path+file_name)
    plt.close()
    #plt.show()

def write_dist(iter, iter2, matrix, directory):
    name = "{0:01d}_{1:06d}".format(iter2,iter)+".txt"
    print(name)
    matrix = np.squeeze(np.array(matrix))
    full_path = directory+name
    f = open(full_path, "w")
    for x in range(matrix.shape[0]):
        for y in range(matrix.shape[1]):
            #print(matrix[x][y])
            f.write("{0:01.6f} ".format(matrix[x][y]))
        f.write("\n")
    f.close()

def write_coordinates(array, matrix_shape, file_name):
    import matplotlib.pyplot as plt
    print(array.shape)
    m = np.zeros(matrix_shape)
    for c in range(0, array.shape[0], 2):
        m[int(array[c])][int(array[(c+1)])] = 1.0
    plt.figure()
    plt.imshow(m.real, cmap='jet', interpolation='none')
    plt.savefig(PATH_OUTPUT+file_name)
    plt.close()
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
    plt.close()

def interpolate(matrix,size):
    matrix = np.array(matrix)
    typ = matrix.dtype
    new_matrix = cv.resize(matrix,(size, size), interpolation = cv.INTER_NEAREST)
    for j in range(new_matrix.shape[0]):
        for i in range(new_matrix.shape[1]):
            if not new_matrix[j][i] >= GHOST_MIN :
                new_matrix[j][i] = np.random.uniform(0.001,0.1)                 # SO that the noise is not interpolated
    return np.array(new_matrix, dtype=typ)

def find_target_for_matrix(y_):
    w_y = 1#np.where(y_ == 1)[1][1]
    h_y = 1#np.where(y_ == 1)[0][1]
    return w_y, h_y

def get_relevant_prediction_index(y_):
    a = []
    for yy_ in y_:
        a.append(find_target_for_matrix(yy_))
    return np.array(a)

def write_matrix(iter,name,matrix):
        matrix = np.array(matrix)
        full_path = EVAL_PATH+name
        f = open(full_path, "w")
        for x in range(matrix.shape[0]):
            for y in range(matrix.shape[1]):
                #print(matrix[x][y])
                f.write("{0:01.6f} ".format(matrix[x][y]))
            f.write("\n")
        f.close()

def generate_evaluation_set(size):
    print("Generate Evaldataset...")
    # save Ground truth
    gt_data = []
    print("Gound Truth...")
    for i in range(size):
        if (i+1) % (size/10)==0:
            print(str(i+1)+"/"+str(size))
        gt, _, _ = read_random_matrix()
        #while gt in gt_data:
            #gt = read_random_matrix()
        #gt = interpolate(gt, ZERO_PADDING_SIZE)
        gt = zero_pad_in(gt)
        gt_data.append(gt)
        file_name = "y_{0:06d}".format(i)+".txt"
        write_matrix(i,file_name,gt)

    # Generate & save input data
    print("Input Data...")
    input_data, _ = random_erase_percentage(gt_data,DIFICULTY_PERCENTAGE)
    for i in range(len(input_data)):
        if (i+1) % (size/10)==0:
            print(str(i+1)+"/"+str(size))
        #input_data = random_erase_percentage(gt_data[i],DIFICULTY_PERCENTAGE)
        #input = interpolate(input, ZERO_PADDING_SIZE)
        #input_data.append(input)
        file_name = "x_{0:06d}".format(i)+".txt"
        write_matrix(i,file_name,input_data[i])

    return input_data, gt_data
'''
def check_zero(matrix):
    if 0.0 in matrix:
        for y in range(matrix.shape[0]):
            for x in range(matrix.shape[1]):
                if matrix[y][x] == 0.0:
                    matrix[y][x] = np.random.uniform(0.001,0.1)
    return matrix
'''

def read_eval_files():
    print("READING EVAL FILES...")
    input_list = list_data(EVAL_PATH,"x")
    gt_list = list_data(EVAL_PATH,"y")

    input_data = []
    gt_data = []

    for x in range(len(input_list)):
        matrix = parse_datafile(input_list[x])
        input_data.append(matrix)

    for y in range(len(gt_list)):
        matrix = parse_datafile(gt_list[y])
        gt_data.append(matrix)

    return input_data, gt_data

def get_evaldata():
    if not os.listdir(EVAL_PATH) :
        print("Directory is empty")
        generate_evaluation_set(EVAL_SIZE)

    input_data, gt_data = read_eval_files()

    return input_data, gt_data




if __name__ == '__main__':
    print("testing compression...")
    #batch = next_comp_batch(1)
    #x_, y_, t_ = next_comp_batch(10)
    #print(x_.shape)
    #print(y_.shape)
    #visualise_mat(x_[0])
    #visualise_mat(y_[0])
    #visualise_compare(x_[0],y_[0])
    _,_ = generate_evaluation_set(EVAL_SIZE)
