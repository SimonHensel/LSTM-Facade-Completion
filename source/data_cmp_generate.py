import numpy as np
import os
import re
import random
import cv2 as cv
from pathlib import Path


PATH_DATA = "data/tests_cluster/"
#PATH_DATA = "data/graz50/graz50_matrix/"
PATH_DISTANCES = "data/Distances/"
PATH_COMPRESSION_INFO = "data/compress_information/"
#PATH_DATA = "/tmp/data/tests_cluster/"
PATH_OUTPUT = "outputs/out_test/"
#PATH_OUTPUT = "/tmp/out_md/"
EVAL_PATH = "evaluation/eval_graz_25/"


EVAL_SIZE = 10000
ZERO_PADDING_SIZE = 25#64#512
DIFICULTY = 8
DIFICULTY_PERCENTAGE = 0.2
DIFICULTY_MAX = 12
DIFICULTY_MIN = 8
GHOST_MIN = 0.4
GHOST_MAX = 0.6
MAX_FALSE = 3
MAX_FALSE_SIZE = 3
MAX_MISSING_COORDINATES = 32

def list_data(input_path, begins_with=None):
    if begins_with == None:
        begins_with='0'
    image_list = []
    file_list = os.listdir(input_path) 
    if file_list != None:
        pass

    for i in range(len(file_list)):
        current_file_abs_path = input_path+file_list[i]
        if (current_file_abs_path.endswith(".txt") and (not current_file_abs_path.startswith('c'))):
            if begins_with == None:
                image_list.append(current_file_abs_path)
            else:
                if (file_list[i].startswith(begins_with)):
                    image_list.append(current_file_abs_path)

    image_list.sort()
    return image_list

def list_dist(input_path, number):
    image_list = []
    file_list = os.listdir(input_path) 
    if file_list != None:
        pass
    for i in range(len(file_list)):
        current_file_abs_path = input_path+file_list[i]
        if (file_list[i].endswith(".txt") and file_list[i].startswith(str(number))):
            image_list.append(current_file_abs_path)

    image_list.sort()
    return image_list

def list_coordinate(input_path, begins_with=None):
    image_list = []
    begins_with = 'c'
    file_list = os.listdir(input_path)
    if file_list != None:
        pass
    for i in range(len(file_list)):
        current_file_abs_path = input_path+file_list[i]
        if (current_file_abs_path.endswith(".txt")):
            if begins_with == None:
                image_list.append(current_file_abs_path)
            else:
                if (file_list[i].startswith(begins_with)):
                    image_list.append(current_file_abs_path)
        else:
            pass
    if len(image_list) != 0:
        for list_index in range(len(image_list)):
            pass
    else:
        pass
    image_list.sort()
    return image_list

def parse_datafile(file_to_read):
    lines = [line.rstrip('\n') for line in open(file_to_read)]

    rows = len(lines)
    cols = len(re.findall("\d+\.\d+", lines[0]))

    matrix = np.random.uniform(low=0.00001, high=0.001, size=(rows,cols))


    for l in range(len(lines)):
        tmp_list = re.findall("\d+\.\d+", lines[l])
        tmp_f_list = [float(i) for i in tmp_list] 
        #if not tmp_f_list[c] == 0.0:
        #    matrix[l][c] = tmp_f_list[c]

    return matrix

def zero_pad_in(matrix):
    matrix = np.array(matrix)
    new_matrix = np.random.uniform(low=0.00001, high=0.001, size=(ZERO_PADDING_SIZE,ZERO_PADDING_SIZE))

    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            new_matrix[y][x] = matrix[y][x]

    return new_matrix


def read_random_matrix():
    data_list = list_data(PATH_DATA)
    coordinate_list = list_coordinate(PATH_DATA)

    #print(len(data_list))

    rand_number = np.random.randint(low=0, high=len(coordinate_list)-1)
    matrix = parse_datafile(data_list[rand_number])
    coordinates_array = read_coordinates(rand_number)

    while max(matrix.shape[0],matrix.shape[1]) > ZERO_PADDING_SIZE:
        rand_number = None
        matrix = None
        rand_number = np.random.randint(low=0, high=len(data_list)-1)
        matrix = parse_datafile(data_list[rand_number])
        coordinates_array = read_coordinates(rand_number)

    return matrix, coordinates_array, rand_number

def read_random_matrix_with_distance():
    data_list = list_data(PATH_DATA)
    coordinate_list = list_coordinate(PATH_DATA)
    dist1_list = list_dist(PATH_DISTANCES,1)
    dist2_list = list_dist(PATH_DISTANCES,2)

    rand_number = np.random.randint(low=0, high=len(coordinate_list)-1)
    matrix = parse_datafile(data_list[rand_number])
    dist1 = parse_datafile(dist1_list[rand_number])
    dist2 = parse_datafile(dist2_list[rand_number])
    coordinates_array = read_coordinates(rand_number)

    while max(matrix.shape[0],matrix.shape[1]) > ZERO_PADDING_SIZE:
        rand_number = None
        matrix = None
        rand_number = np.random.randint(low=0, high=len(data_list)-1)
        matrix = parse_datafile(data_list[rand_number])
        dist1 = parse_datafile(dist1_list[rand_number])
        dist2 = parse_datafile(dist2_list[rand_number])

    return matrix, dist1, dist2



def count_objects(matrix):
    gray = np.array(matrix+1,dtype=np.uint8)
    ret, thresh = cv.threshold(gray, 1, 1, 1)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return len(contours)

def make_predonly_y(matrix_list, perc):
    new_matrix_list = []
    y_matrix_list = []
    for x in range(len(matrix_list)):
        matrix = matrix_list[x]
        number_of_objects = int(count_objects(matrix)*perc)

        y_len = matrix.shape[0]
        x_len = matrix.shape[1]

        new_matrix = matrix.copy()
        y_matrix = np.random.uniform(0.0001,0.001,size=matrix.shape)

        for i in range(number_of_objects):
            selected_value = 0
            while not selected_value == 1.0:
                x_rand = np.random.randint(low=0, high=x_len)
                y_rand = np.random.randint(low=0, high=y_len)

                selected_value = matrix[y_rand][x_rand]

            neighbors = []
            neighbors.append([y_rand,x_rand])

            while neighbors :
                y,x = neighbors[len(neighbors)-1]
                neighbors.pop()
                if  y < y_len-1 and new_matrix[y+1][x] == 1.0:
                    neighbors.append([y+1,x])
                if x < x_len-1 and new_matrix[y][x+1] == 1.0:
                    neighbors.append([y,x+1])
                if y > 0 and new_matrix[y-1][x] == 1.0:
                    neighbors.append([y-1,x])
                if x > 0 and new_matrix[y][x-1] == 1.0:
                    neighbors.append([y,x-1])

                new_matrix[y][x] = np.random.uniform(0.001,0.1)
                y_matrix[y][x] = 1.0

        new_matrix_list.append(new_matrix)
        y_matrix_list.append(y_matrix)

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

        y_len = matrix.shape[0]
        x_len = matrix.shape[1]

        new_matrix = matrix.copy()

        coords = [(x,y) for x in range(x_len) for y in range(y_len)]
        random.shuffle(coords)

        erased_objects_part = []
        counter = 0
        iter = 0

        while iter < len(coords) and counter < number_of_objects:
            y_iter = coords[iter][1]
            x_iter = coords[iter][0]
            if new_matrix[y_iter][x_iter] > 0.9:
                counter += 1

                neighbors = []
                neighbors.append([y_iter,x_iter])

                while neighbors :
                    y,x = neighbors[len(neighbors)-1]
                    neighbors.pop()

                    if  y < y_len-1 and new_matrix[y+1][x] > 0.9:
                        neighbors.append([y+1,x])
                    if x < x_len-1 and new_matrix[y][x+1] > 0.9:
                        neighbors.append([y,x+1])
                    if y > 0 and new_matrix[y-1][x] > 0.9:
                        neighbors.append([y-1,x])
                    if x > 0 and new_matrix[y][x-1] > 0.9:
                        neighbors.append([y,x-1])

                    new_matrix[y][x] = np.random.uniform(0.00001,0.001)
                    if coordinate_array:
                        erased_objects_part.append([float(x),float(y)])
            iter += 1





        while len(erased_objects_part) < MAX_MISSING_COORDINATES and coordinate_array: 
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

        erased_objects = np.array(erased_objects)
        erased_objects = erased_objects.reshape((erased_objects.shape[0],erased_objects.shape[1]*erased_objects.shape[2]))

    return new_matrix_list, erased_objects

def random_ghosting_erase(matrix_list, perc, coordinate_array=None):
    new_matrix_list = []
    erased_objects = []
    for x in range(len(matrix_list)):
        matrix = matrix_list[x]
        number_of_objects = int(count_objects(matrix)*perc)
        if number_of_objects < 1:
            number_of_objects = 1

        if number_of_objects > (MAX_MISSING_COORDINATES/2):
            number_of_objects = int(MAX_MISSING_COORDINATES/2)

        y_len = matrix.shape[0]
        x_len = matrix.shape[1]

        new_matrix = matrix.copy()

        for i in range(number_of_objects):
            erased_objects_part = []
            ghost_value = np.random.uniform(0.0 ,GHOST_MAX)

            selected_value = 0
            while not selected_value > 0.9:
                x_rand = np.random.randint(low=0, high=x_len)
                y_rand = np.random.randint(low=0, high=y_len)

                selected_value = matrix[y_rand][x_rand]

            neighbors = []
            neighbors.append([y_rand,x_rand])

            while neighbors :
                y,x = neighbors[len(neighbors)-1]
                neighbors.pop()
                if  y < y_len-1 and new_matrix[y+1][x] > 0.9:
                    neighbors.append([y+1,x])
                if x < x_len-1 and new_matrix[y][x+1] > 0.9:
                    neighbors.append([y,x+1])
                if y > 0 and new_matrix[y-1][x] > 0.9:
                    neighbors.append([y-1,x])
                if x > 0 and new_matrix[y][x-1] > 0.9:
                    neighbors.append([y,x-1])

                if ghost_value <= GHOST_MIN:
                    new_matrix[y][x] = np.random.uniform(0.001,0.1)
                else:
                    new_matrix[y][x] = ghost_value
                if coordinate_array:
                    erased_objects_part.append([float(x),float(y)])

        while len(erased_objects_part) < MAX_MISSING_COORDINATES and coordinate_array: 
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

                tmp_ghost_value = np.random.uniform(GHOST_MIN,GHOST_MAX)

                new_matrix[y_min:y_max,x_min:x_max] = tmp_ghost_value

        erased_objects.append(erased_objects_part)
        new_matrix_list.append(new_matrix)

    if coordinate_array:
        for o in range(len(erased_objects)):
            erased_objects[o]= coordinate_sort(erased_objects[o])
            erased_objects[o] = np.array(erased_objects[o])

        erased_objects = np.array(erased_objects)
        erased_objects = erased_objects.reshape((erased_objects.shape[0],erased_objects.shape[1]*erased_objects.shape[2]))

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

    rows = len(lines)
    cols = 2

    for l in range(len(lines)):
        tmp_list = re.findall("\d+\.\d+", lines[l])
        for c in range(len(tmp_list)):
            coordinate_array.append(tmp_list[c])

    return np.array(coordinate_array)

def random_rotate(matrix):
    rand_num = random.randint(0,3)
    for i in range(rand_num):
        matrix = np.rot90(matrix)

    return matrix

def random_flip(matrix):
    do_it = random.randint(0,1)
    hoz = random.randint(0,1)

    if do_it == 1:
        if hoz==1:
            out = np.flip(matrix, 1)
        else:
            out = np.flip(matrix, 0)
    else:
        out = matrix

    return out

def next_batch(bs):
    x = []
    y = []
    y_tmp = []
    for i in range(bs):
        random_matrix, _ , _ = read_random_matrix()
        y.append(np.copy(random_matrix))
        clusters = hoz_clustering(random_matrix)

        for c in range(len(clusters)):
            cluster = clusters[c]
            for p in range(len(cluster)):
                random_matrix[cluster[p][0],cluster[p][1]] = float(c+1)
        y_tmp.append(random_matrix)

    x_matrix, _ = random_erase_percentage(y_tmp,DIFICULTY_PERCENTAGE)
    x_array = np.array(x_matrix)

    y_array = np.array(y)

    for i in range(bs):
        x.append(normalize_cluster(zero_pad_in(x_array[i])))
        y[i] = zero_pad_in(y_array[i])

    x = np.array(x)
    y = np.array(y)
    if np.any(np.isin(x,0.0)) or np.any(np.isin(y,0.0)):
        print("ERROR: Zero found!")
        exit()
    return x, y
"""
def compress(batches, numbers):
    #Compresses the Matrices according to the compression files
    batches = np.array(batches)
    new_batch = []

    for i in range(len(numbers)):
        infile = PATH_COMPRESSION_INFO+"{0:06d}".format(numbers[i])+".txt"

        information = open(infile)
        lines = [line.rstrip('\n') for line in information]
        rows = len(lines)
        if not rows == 2:
            print("Error: to much lines - "+ str(rows))
        y_len = len(re.findall("\d+", lines[0]))
        x_len = len(re.findall("\d+", lines[1]))

        y_info = list(map(int,re.findall("\d+", lines[0])))
        x_info = list(map(int,re.findall("\d+", lines[1])))
        information.close()

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

    return np.array(new_batch)
"""
"""
def next_comp_batch(bs):
    x = []
    y = []
    y_tmp = []
    numbers = []
    for i in range(bs):
        random_matrix, _, rand_number= read_random_matrix()
        y_tmp.append(random_matrix)
        numbers.append(rand_number)

    x_matrix, _ = random_erase_percentage(y_tmp,DIFICULTY_PERCENTAGE)
    y_tmp = compress(y_tmp, numbers)
    x_matrix = compress(x_matrix, numbers)
    x_array = np.asarray(x_matrix)

    y_array = np.asarray(y_tmp)

    for i in range(bs):
        x.append(zero_pad_in(x_array[i]))
        y.append(zero_pad_in(y_array[i]))

    t = get_relevant_prediction_index(y)
    x = np.array(x)
    y = np.array(y)
    if np.any(np.isin(x,0.0)) or np.any(np.isin(y,0.0)):
        print("ERROR: Zero found!")
        exit()

    return x, y, t
"""

def visualise_mat(m):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(m.real, cmap='jet', interpolation='none')
    plt.show()
    plt.close()

def write_mat(m, file_name):
    import matplotlib.pyplot as plt
    Path(PATH_OUTPUT).mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.imshow(m.real, cmap='jet', interpolation='none')
    plt.savefig(PATH_OUTPUT+file_name)
    plt.close()
    #plt.show()

def write_mat_conf(m, file_name, path):
    import matplotlib.pyplot as plt
    Path(path).mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.imshow(m.real, cmap='jet', interpolation='none')
    plt.savefig(path+file_name)
    plt.close()
    #plt.show()

def write_dist(iter, iter2, matrix, directory):
    name = "{0:01d}_{1:06d}".format(iter2,iter)+".txt"
    #print(name)
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
    Path(PATH_OUTPUT).mkdir(parents=True, exist_ok=True)
    #print(array.shape)
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
        gt, _, _ = read_random_matrix()

        gt = zero_pad_in(gt)
        gt_data.append(gt)
        file_name = "y_{0:06d}".format(i)+".txt"
        write_matrix(i,file_name,gt)

    # Generate & save input data
    print("Input Data...")
    input_data, _ = random_erase_percentage(gt_data,DIFICULTY_PERCENTAGE)
    for i in range(len(input_data)):
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
        clusters = np.array(hoz_clustering(matrix)) #NEW
        for c in range(len(clusters)):
            cluster = clusters[c]
            for p in range(len(cluster)):
                matrix[cluster[p][0],cluster[p][1]] = float(c+1)
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

def reverse_matrix(matrix):
    rmatrix = np.copy(matrix)

    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            if rmatrix[y,x] == 1.0:
                rmatrix[y,x] = 0.0
            else:
                rmatrix[y,x] = 1.0
    return rmatrix

def regionQuery(matrix,point,eps):
    rquery = []
    y_co = point[0]
    x_co = point[1]

    for y in range(max(y_co-eps,0), min(y_co+eps+1,matrix.shape[0])):
        for x in range(max(x_co-eps,0),min(x_co+eps+1,matrix.shape[1])):
            if matrix[y,x] == 1.0: #and not([y,x] == point):
                rquery.append([y,x])

    return rquery

def get_points(matrix):
    points = []
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            if matrix[y,x] == 1.0:
                points.append([y,x])
    return points

def expandCluster(matrix,point, visited, points, all_clusters ,cluster, eps, MinPts):
    """
    expandCluster(P, N, C, eps, MinPts)
        add P to cluster C
        for each point P' in N
            if P' is not visited
                mark P' as visited
                N' = D.regionQuery(P', eps)
                if sizeof(N') >= MinPts
                    N = N joined with N'
            if P' is not yet member of any cluster
                add P' to cluster C
                unmark P' as NOISE if necessary
    """
    cluster.append(point)
    while (len(points) > 0):
        p = points.pop()
        p = np.array(p)
        if visited[p[0],p[1]] == 0.0:
            visited[p[0],p[1]] = 1.0

            new_points = regionQuery(matrix,p,eps)

            if len(new_points) >= MinPts:
                points = points+new_points
        p_in = 0
        for i in range(len(all_clusters)):
            if np.all(all_clusters[i] == p):
                p_in = 1

        if p_in == 0:
            cluster.append(p)

def matrix_dbscan(matrix, eps, MinPts):
    current = 0
    clusters = []
    all_noise = []
    all_points = get_points(matrix)
    visited = reverse_matrix(matrix) #visited matrix
    #visualise_mat(matrix)
    #visualise_mat(visited)
    all_points = np.array(all_points)


    #for p in range(len(points)):
    while( not np.all(visited == 1)):
        if visited[all_points[current,0],all_points[current,1]] == 0.0:
            visited[all_points[current,0],all_points[current,1]] = 1.0
            points = regionQuery(matrix,all_points[current],eps)
            if len(points) < MinPts:
                all_noise.append(all_points[current])
            else:
                new_cluster = []
                expandCluster(matrix, all_points[current], visited, points, clusters, new_cluster, eps, MinPts)
                clusters.append(new_cluster)

        current += 1
        current = current%len(all_points)
    for c in range(len(clusters)):
        #print(clusters[c].shape)
        clusters[c] = np.unique(clusters[c], axis=0)

    return clusters, all_noise

def get_all_y(cluster):
    cluster = np.array(cluster)

    all_y = np.unique(cluster[:,0])
    np.sort(all_y)

    return all_y

def get_all_x(cluster):
    cluster = np.array(cluster)

    all_x = np.unique(cluster[:,1])
    np.sort(all_x)

    return all_x

def vert_clustering(matrix):
    """

    """
    first_clustering, _ = matrix_dbscan(matrix,1,1)
    matrix2 = np.zeros(matrix.shape)

    #print("########FIRST CLUSTERING############")
    #print(first_clustering)
    #print("####################################")

    for c in range(len(first_clustering)):
        cluster = first_clustering[c]
        #print(" ")
        for p in range(len(cluster)):
            matrix2[cluster[p][0],cluster[p][1]] = float(c+1)

    result = []
    already_clustered = np.zeros((len(first_clustering)))

    for c in range(len(first_clustering)):#while (c < len(first_clustering)):
        #print(already_clustered)
        if not already_clustered[c] == 1.0:
            new_cluster = []
            new_cluster.extend(first_clustering[c])
            already_clustered[c] = 1.0
            for c2 in range(len(first_clustering)):#while (c2 < len(first_clustering)):
                #print("c:"+str(c)+" c2:"+str(c2))

                tmp1 = np.array(first_clustering[c])
                tmp2 = np.array(first_clustering[c2])

                all_x1 = get_all_x(tmp1)
                all_x2 = get_all_x(tmp2)

                #print("all_x1: "+str(all_x1)+" all_x2: "+str(all_x2))

                if tmp1.shape[0] == tmp2.shape[0] and not(already_clustered[c2] == 1.0) and not c == c2:
                    if np.all(all_x1 == all_x2):
                        #print("are same")
                        new_cluster.extend(first_clustering[c2])
                        already_clustered[c2] = 1.0
                        #first_clustering[c].extend(first_clustering[c2])
                        #first_clustering.pop(c2)
                c2 += 1
        #print(new_cluster)
        result.append(new_cluster)
        c += 1
    return result


def hoz_clustering(matrix):
    """

    """
    first_clustering, _ = matrix_dbscan(matrix,1,1)
    matrix2 = np.zeros(matrix.shape)

    for c in range(len(first_clustering)):
        cluster = first_clustering[c]
        #print(" ")
        for p in range(len(cluster)):
            matrix2[cluster[p][0],cluster[p][1]] = float(c+1)
    """
    for y in range(matrix2.shape[0]):
        for x in range(matrix2.shape[1]):
            if
    """

    result = []
    already_clustered = np.zeros((len(first_clustering)))

    for c in range(len(first_clustering)):#while (c < len(first_clustering)):
        #print(already_clustered)
        if not already_clustered[c] == 1.0:
            new_cluster = []
            new_cluster.extend(first_clustering[c])
            already_clustered[c] = 1.0
            for c2 in range(len(first_clustering)):#while (c2 < len(first_clustering)):
                #print("c:"+str(c)+" c2:"+str(c2))

                tmp1 = np.array(first_clustering[c])
                tmp2 = np.array(first_clustering[c2])

                all_y1 = get_all_y(tmp1)
                all_y2 = get_all_y(tmp2)

                #print("all_y1: "+str(all_y1)+" all_y2: "+str(all_y2))

                if tmp1.shape[0] == tmp2.shape[0] and not(already_clustered[c2] == 1.0) and not c == c2:
                    if np.all(all_y1 == all_y2):
                        #print("are same")
                        new_cluster.extend(first_clustering[c2])
                        already_clustered[c2] = 1.0
                        #first_clustering[c].extend(first_clustering[c2])
                        #first_clustering.pop(c2)
                c2 += 1
        #print(new_cluster)
        result.append(new_cluster)
        c += 1
    #print(len(result))
    #print(len(result[0]))
    #print(len(result[1]))
    #print(result)

    return np.array(result)

def normalize_cluster(matrix):
    new_matrix = np.copy(matrix)

    mapping = np.unique(matrix.astype(np.int64))

    #mapping = np.delete(mapping,0)
    for y in range(new_matrix.shape[0]):
        for x in range(new_matrix.shape[1]):
            if (new_matrix[y][x]> 0.9):
                #print(mapping)
                #print("matrix[y][x]: "+str(matrix[y][x]))
                #print("int(matrix[y][x]): "+str(np.where(mapping==(int(new_matrix[y][x])))[0]))
                new_matrix[y][x] = float(np.where(mapping==(int(new_matrix[y][x])))[0])

    return new_matrix

def visualise_clusters(matrix, clusters):
    import matplotlib.pyplot as plt
    columns = 2
    rows = 1

    matrix2 = np.zeros(matrix.shape)
    for c in range(len(clusters)):
        cluster = clusters[c]
        #print(" ")
        for p in range(len(cluster)):
            #print(p)
            #print(str(cluster[p][0])+" "+str(cluster[p][1]))
            #print()
            matrix2[cluster[p][0],cluster[p][1]] = float(c+1)

    plt.close('all')
    plt.switch_backend('GTK3Agg') 
    fig = plt.figure()
    #print("done")
    ax = fig.add_subplot(rows, columns, 1)
    plt.imshow(matrix, cmap='viridis', interpolation='none')
    ax = fig.add_subplot(rows, columns, 2)

    for y in range(matrix2.shape[0]):
        for x in range(matrix2.shape[1]):
            label = matrix2[y, x]
            text_x = x
            text_y = y
            ax.text(text_x, text_y, label, color='black', ha='center', va='center')

    plt.imshow(matrix2, cmap='viridis', interpolation='none')
    plt.show()
    plt.close()

def test_regionQuery():#(matrix,point,eps)
    matrix = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

    matrix = np.array(matrix)

    result = regionQuery(matrix,[3,3],1)
    print("regionQuery(matrix,[1,1],1)")
    print(result)

    result = regionQuery(matrix,[3,3],2)
    print("regionQuery(matrix,[1,1],2)")
    print(result)

    result = regionQuery(matrix,[3,3],3)
    print("regionQuery(matrix,[1,1],3)")
    result = np.array(result)

    print(np.array(result))





def test_clustering():
    matrix, _, _ = read_random_matrix()
    matrix = np.array(matrix)
    print("DBSCAN")
    dbscan_clusters, _ = matrix_dbscan(matrix, 2, 2) #matrix_dbscan(matrix, eps, MinPts)
    print("DBSCAN + Horizontal")
    hoz_clusters = np.array(hoz_clustering(matrix))
    print("DBSCAN + Vertical")
    vert_clusters = np.array(vert_clustering(matrix))
    print("###########RESULT############")
    #clusters = np.array(clusters)
    #print(clusters.shape)
    #print(clusters)
    visualise_clusters(matrix,dbscan_clusters)
    print("###########RESULT############")
    visualise_clusters(matrix,hoz_clusters)
    print("###########RESULT############")
    visualise_clusters(matrix,vert_clusters)


def test_normalize_cluster():
    matrix = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 3.0, 3.0, 3.0, 0.0, 7.0, 7.0, 7.0, 0.0],
        [0.0, 3.0, 3.0, 3.0, 0.0, 7.0, 7.0, 7.0, 0.0],
        [0.0, 3.0, 3.0, 3.0, 0.0, 7.0, 7.0, 7.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    print(matrix)

    new_matrix = normalize_cluster(matrix)
    print(new_matrix)

    visualise_compare(matrix,new_matrix)



def test_size(size_m):
    count = 0
    data_list = list_data(PATH_DATA)
    print(len(data_list))

    for i in range(len(data_list)):
        matrix = parse_datafile(data_list[i])
        if matrix.shape[0] <= size_m and matrix.shape[1] <= size_m:
            count += 1

    percentage = count/len(data_list)
    print()

    return count



if (__name__ == '__main__'):
    #print("testing compression...")
    #x_, y_, t_ = next_batch(1)
    #print(batch.shape)
    #x_, y_, t_ = next_comp_batch(10)
    #print(batch)
    #print(x_.shape)
    #print(y_.shape)
    #visualise_mat(x_[0])
    #visualise_mat(y_[0])
    #visualise_compare(x_[0],y_[0])
    #_,_ = generate_evaluation_set(EVAL_SIZE)
    #print("Testing clustering...")
    #test_regionQuery()
    #test_clustering()
    #test_normalize_cluster()
    s = test_size(25)

    print(s)
