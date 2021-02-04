import cv2 as cv
import numpy as np
import os
import random
##################################################################
'''
Generates a dataset with a wanted size through
removing Objects from the original CMP Datasetself.
Since can remove objects at random the dataset nearly
can have any size we want.
'''
#################################################################

PATH_BASEDATASET = "data/graz50/graz50_facade_dataset/"
IMAGESETS_SEG_DIR = "labels_used/"

PATH_RNN_DATASET = "data/graz50/"
RNN_TEST_DIR = "graz50_matrix/"
COMP_INFO_DIR = "compress_information/"

NUMBER_OF_OBJECTS = 1
MAX_SIZE = 64
MAX_THRESHOLD = 8
CLUSTER_SIZE = 64


to_delete = []

MAX_MAT_X = 0
MAX_MAT_Y = 0

MIN_MAT_X = 99999
MIN_MAT_Y = 99999

'''
def cluster_gridlines(grid):
    grid_x = grid[0]
    grid_y = grid[1]
    if len(grid_x) <= CLUSTER_SIZE and len(grid_x) <= CLUSTER_SIZE:
        #print("NO clustering needed...")
        return [grid_x,grid_y]

    #print("Clustering operation...")

    threshold = 1
    new_grid_x = []
    new_grid_y = []

    while (len(new_grid_x) > CLUSTER_SIZE or len(new_grid_x) == 0) and threshold < MAX_THRESHOLD:
        #print("LEN "+str(len(grid_x)))
        x = 0
        new_grid_x.clear()
        #for x in range(0,(len(grid_x)-1)):
        while x < (len(grid_x)-1):
            if (grid_x[x+1]-grid_x[x]) <= threshold:
                tmp_value = int((grid_x[x+1]+grid_x[x])/2)
                if tmp_value in new_grid_x:
                    print("ERROR!")
                new_grid_x.append(tmp_value)
                x += 2
            else:
                if x == (len(grid_x)-2):
                    new_grid_x.append(grid_x[x+1])
                new_grid_x.append(grid_x[x])
                x += 1
        #print(str(threshold)+"- New LEN:"+str(len(new_grid_x)))
        threshold += 1

    if threshold >= MAX_THRESHOLD:
        #print("WARNING: Couldnt Cluster to wished size!")
        return [grid_x,grid_y]
    threshold = 1

    while (len(new_grid_y) > CLUSTER_SIZE  or len(new_grid_y) == 0) and threshold < MAX_THRESHOLD:
        y = 0
        new_grid_y.clear()
        while y < (len(grid_y)-1):
            if grid_y[y+1]-grid_y[y] <= threshold:
                tmp_value = int((grid_y[y+1]+grid_y[y])/2)
                if tmp_value in new_grid_y:
                    print("ERROR!")
                new_grid_y.append(tmp_value)
                y += 2
            else:
                if y == (len(grid_y)-2):
                    new_grid_y.append(grid_y[y+1])
                new_grid_y.append(grid_y[y])
                y += 1
        #print(str(threshold)+"- New LEN:"+str(len(new_grid_y)))
        threshold += 1
    if threshold >= MAX_THRESHOLD:
        #print("WARNING: Couldnt Cluster to wished size!")
        return [grid_x,grid_y]

    return [new_grid_x,new_grid_y]
'''

def image_to_grid(img):
    y_size = img.shape[0]
    x_size = img.shape[1]
    #Following colorsets are only valid for the GRaz50 facade dataset
    color_objects = [[0,0,255],[0,128,255]]#, [255,0,128 # window,door,balcony
    backgrounds = [[255,255,128],[0,255,255],[0,0,0]] # air, wall, black
    color_objects = np.array(color_objects)
    backgrounds = np.array(backgrounds)

    x_coordinates = []
    y_coordinates = []

    x_coordinates.append(0)
    y_coordinates.append(0)


    for y in range(1,y_size-1):
        for x in range(1,x_size-1):
            current_color = img[y,x]
            left = img[y,x-1]
            right = img[y,x+1]
            up = img[y+1,x]
            down = img[y-1,x]

            """
            print("\n###########")
            print(current_color)
            print(left)
            print(right)
            print(up)
            print(down)
            print("")
            print(color_objects[0])
            print(color_objects[1])
            print(backgrounds[0])
            print(backgrounds[1])
            print("\n#############")
            """

            if ((np.all(current_color == color_objects[0]) or np.all(current_color == color_objects[1])) and
                    (np.all(left == backgrounds[0]) or np.all(left == backgrounds[1]) or np.all(left == backgrounds[2]))):
                #print(str(current_color)+" = "+str(color_objects)+" and "+str(left)+" = "+str(backgrounds))
                if not x in x_coordinates:
                    x_coordinates.append(x)
            elif ((np.all(current_color == color_objects[0]) or np.all(current_color == color_objects[1])) and
                    (np.all(right == backgrounds[0]) or np.all(right == backgrounds[1]) or np.all(right == backgrounds[2]))):
                if not x in x_coordinates:
                    x_coordinates.append(x)
            else:
                pass

            if ((np.all(current_color == color_objects[0]) or np.all(current_color == color_objects[1])) and
                    (np.all(up == backgrounds[0]) or np.all(up == backgrounds[1]) or np.all(up == backgrounds[2]))):
                if not y in y_coordinates:
                    y_coordinates.append(y)
            elif ((np.all(current_color == color_objects[0]) or np.all(current_color == color_objects[1])) and
                    (np.all(down == backgrounds[0]) or np.all(down == backgrounds[1]) or np.all(down == backgrounds[2]))):
                if not y in y_coordinates:
                    y_coordinates.append(y)
            else:
                pass

    x_coordinates.append(x_size-1)
    y_coordinates.append(y_size-1)

    x_coordinates.sort()
    y_coordinates.sort()

    print("image_to_grid out:")
    print("Max:"+str(y_size*x_size))
    print(y_coordinates)
    print(x_coordinates)


    return y_coordinates, x_coordinates


def paint_grid(img, y_coordinates, x_coordinates):
    color = [255,255,255]

    grid_img = np.copy(img)

    for x in range(len(x_coordinates)):
        for y in range(grid_img.shape[0]):
            #print(img.shape)
            #print(x_grid[x])
            grid_img[y][x_grid[x]] = color

    for y in range(len(y_coordinates)):
        for x in range(grid_img.shape[1]):
            #print(y_grid[y])
            grid_img[y_grid[y]][x] = color

    return grid_img


def compress_matrix(matrix):
    x_len = matrix.shape[1]
    y_len = matrix.shape[0]
    new_matrix = np.copy(matrix)
    x_comp_info = []
    y_comp_info = []
    counter = 0

    for y in range(y_len):
        if not np.any(matrix[y,:] > 0.5):
            new_matrix = np.delete(new_matrix,y-counter,0)
            y_comp_info.append(y)
            counter += 1

    counter = 0

    for x in range(x_len):
        if not np.any(matrix[:,x] > 0.5):
            new_matrix = np.delete(new_matrix,x-counter,1)
            x_comp_info.append(x)
            counter += 1

    return new_matrix, np.array(x_comp_info), np.array(y_comp_info)


def write_matrix_comp(iter,matrix, x_comp_info, y_comp_info):
    if matrix.shape[0] < MAX_SIZE and matrix.shape[1] < MAX_SIZE:
        test_file = "{0:06d}".format(iter)+".txt"
        full_path = PATH_RNN_DATASET+RNN_TEST_DIR+test_file
        comp_full_path = PATH_RNN_DATASET+COMP_INFO_DIR+test_file
        #Write Matrix file
        f = open(full_path, "w")

        for x in range(matrix.shape[0]):
            for y in range(matrix.shape[1]):
                f.write(str(matrix[x][y])+" ")
            f.write("\n")
        f.close()
        #Write Compressing information
        fc = open(comp_full_path, "w")
        fc.write("y ")
        for yc in range(y_comp_info.shape[0]):
            fc.write(str(y_comp_info[yc])+" ")
        fc.write("\nx ")
        for xc in range(x_comp_info.shape[0]):
            fc.write(str(x_comp_info[xc])+" ")
        fc.close()
    else:
        to_delete.append(iter)





def update_min_max(matrix):
    global MAX_MAT_X
    global MAX_MAT_Y
    global MIN_MAT_X
    global MIN_MAT_Y
    x_len = matrix.shape[0]
    y_len = matrix.shape[1]

    if x_len > MAX_MAT_X :
        MAX_MAT_X = x_len
    if y_len > MAX_MAT_Y :
        MAX_MAT_Y = y_len
    if x_len < MIN_MAT_X :
        MIN_MAT_X = x_len
    if y_len < MIN_MAT_Y :
        MIN_MAT_Y = y_len


def write_matrix(iter,matrix):
    if matrix.shape[0] < MAX_SIZE and matrix.shape[1] < MAX_SIZE:
        test_file = "{0:06d}".format(iter)+".txt"
        full_path = PATH_RNN_DATASET+RNN_TEST_DIR+test_file

        f = open(full_path, "w")

        for x in range(matrix.shape[0]):
            for y in range(matrix.shape[1]):
                f.write(str(matrix[x][y])+" ")
            f.write("\n")

        f.close()
    else:
        to_delete.append(iter)

def add_grid_lines(x_grid, y_grid, img):
    print(img.shape)
    print(x_grid[len(x_grid)-1])
    print(y_grid[len(y_grid)-1])

    grid_img = np.copy(img)

    for x in range(len(x_grid)):
        for y in range(grid_img.shape[0]):
            #print(img.shape)
            #print(x_grid[x])
            grid_img[y][x_grid[x]] = [255,255,255]

    for y in range(len(y_grid)):
        for x in range(grid_img.shape[1]):
            #print(y_grid[y])
            grid_img[y_grid[y]][x] = [255,255,255]

    return grid_img


def list_xml(input_path):
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
        if (current_file_abs_path.endswith(".xml")):
            image_list.append(current_file_abs_path)
            #print("Found %s successfully!" % current_file_abs_path)
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
    return image_list

def list_imgs(input_path):
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
        if (current_file_abs_path.endswith(".png")):
            image_list.append(current_file_abs_path)
            #print("Found %s successfully!" % current_file_abs_path)
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
    return image_list

def parse_xml(xml_file): # reuse annotation file for additional data
    #ano_path = MAIN_PATH+ANNOTATION_DIR+"{0:06d}".format(it_from)+".xml"
    bboxes = []
    bbox = []

    #to_path = MAIN_PATH+ANNOTATION_DIR+"{0:06d}".format(it_to)+".xml"
    #to_img = cv.imread(MAIN_PATH+IMAGE_DIR+"{0:06d}".format(it_to)+".jpg")
    #to_height,to_width = to_img.shape[:2]

    #print(ano_path)
    #print(xml_file)


    with open(xml_file, "r") as f:
        lines = f.readlines()

    #print(lines)
    i = 0
    while i < len(lines):
        if "name" in lines[i]: #damit nur fenster benutzt werden
            if not ("window" in lines[i] or "door" in lines[i]):
                print("Other Object!")
                i += 11
            #else:
                #print(lines[i])
        if "xmin" in lines[i]:
            #print(int(lines[i][lines[i].find(">")+1:lines[i].find("</")]))
            bbox.append(int(lines[i][lines[i].find(">")+1:lines[i].find("</")]))

        if "ymin" in lines[i]:
            #print(int(lines[i][lines[i].find(">")+1:lines[i].find("</")]))
            bbox.append(int(lines[i][lines[i].find(">")+1:lines[i].find("</")]))

        if "xmax" in lines[i]:
            #print(int(lines[i][lines[i].find(">")+1:lines[i].find("</")]))
            bbox.append(int(lines[i][lines[i].find(">")+1:lines[i].find("</")]))

        if "ymax" in lines[i]:
            #print(int(lines[i][lines[i].find(">")+1:lines[i].find("</")]))
            bbox.append(int(lines[i][lines[i].find(">")+1:lines[i].find("</")]))
            bboxes.append(bbox)
            bbox = []

        i += 1

    return bboxes

def remove_random_object(bboxes,number):
    for i in range(number):
        bboxes.pop(random.randint(0,len(bboxes)-1))
    return number

def extract_grids(bboxes, images):
    if len(bboxes) == len(images):
        all_grids = []
        for i in range(len(bboxes)):
            boxes = bboxes[i]
            image = images[i]

            grid_x = []
            grid_y = []

            #add image boarder
            grid_x.append(0)
            grid_y.append(0)

            grid_x.append(image.shape[1]-1)
            grid_y.append(image.shape[0]-1)

            for b in range(len(boxes)):
                if not int(boxes[b][0]) in grid_x :
                    grid_x.append(int(boxes[b][0]))

                if not int(boxes[b][2]) in grid_x :
                    grid_x.append(int(boxes[b][2]))

                if not int(boxes[b][1]) in grid_y :
                    grid_y.append(int(boxes[b][1]))

                if not int(boxes[b][3]) in grid_y :
                    grid_y.append(int(boxes[b][3]))

            grid_x.sort()
            grid_y.sort()

            while grid_x[len(grid_x)-1] >= image.shape[1]:
                print("WARNING: Gridline position OutofBound")
                print(grid_x[len(grid_x)-1])
                print(image.shape[1])
                grid_x[len(grid_x)-1] -= 1

            while grid_y[len(grid_y)-1] >= image.shape[0]:
                print("WARNING: Gridline position OutofBound")
                print(grid_y[len(grid_y)-1])
                print(image.shape[0])
                grid_y[len(grid_y)-1] -= 1



            all_grids.append([grid_x,grid_y])

        return all_grids

    else:
        print("Extract Grids")
        print("ERROR: Number of Annotations and Images differs.")
        print("bboxes: "+ str(len(bboxes)))
        print("images: "+ str(len(images)))
        return [] , []

def reconstruct_labelimage(x_grid, y_grid, matrix, image):
    label_image = np.copy(image)

    for colums in range(matrix.shape[1]):
        for rows in range(matrix.shape[0]):
            if matrix[rows][colums] == 1.0 :
                cv.rectangle(label_image,(x_grid[colums],y_grid[rows]),(x_grid[colums+1],y_grid[rows+1]),(255,255,255),3)

    return label_image

'''
def reconstruct_labelimage(x_grid, y_grid, matrix, image):
    #newimg = add_grid_lines(x_grid, y_grid, image)
    label_image = add_grid_lines(x_grid, y_grid, image)
    #newX = newimg.shape[1]
    #newY = newimg.shape[0]
    #label_image = cv.resize(newimg,(int(newX),int(newY)))
    print(matrix.shape)
    print(len(y_grid))
    print(len(x_grid))


    for columns in range(0,len(x_grid)-3): #ohne rand
        for rows in range(0,len(y_grid)-3): #ohne rand

            if matrix[rows][columns] == 1.0 :
                #cv.rectangle(label_image,(x_grid[colums],y_grid[rows]),(x_grid[colums+1],y_grid[rows+1]),(0,0,255),3)
                x_text = x_grid[columns] #(x_grid[colums]+x_grid[colums+1])/2
                y_text = y_grid[rows] #(y_grid[rows]+y_grid[rows+1])/2
                font                   = cv.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (x_text,y_text)
                fontScale              = 0.5
                fontColor              = (255,255,255)
                lineType               = 1
                cv.putText(label_image,str(int(matrix[rows][columns])),
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)

            #cv.imshow(str(y_text)+","+str(x_text),label_image)
            #cv.waitKey(0)
            #cv.destroyAllWindows()

    return label_image
'''
def set_object_alt(label_image, matrix, x_grid, y_grid):
    color_objects = [[0,0,255],[0,128,255]]
    print(x_grid)
    print(y_grid)

    for x_g in range(len(x_grid)-1):
        for y_g in range(len(y_grid)-1):
            #box = bbox[i]
            #mid_x = (box[0]+box[2])/2#(x_grid[x]+x_grid[x+1])/2
            #mid_y = (box[1]+box[3])/2#(y_grid[y]+y_grid[y+1])/2

            #if a point of the bounding box is in the part of the grid it belongs to an object
            #GOING OVER PIXELS OF A CELL
            founds = 0
            count = 0
            print("+++++++++++++++++++++")
            print("x_grid:"+str(x_grid[x_g])+" -> "+str(x_grid[x_g+1]))
            print("y_grid:"+str(y_grid[y_g])+" -> "+str(y_grid[y_g+1]))
            for b_x in range(x_grid[x_g],x_grid[x_g+1]-1):
                for b_y in range(y_grid[y_g],y_grid[y_g+1]-1):
                    current_color = label_image[b_y,b_x]
                    if (np.all(current_color == color_objects[0]) or
                            np.all(current_color == color_objects[1])):
                        founds = founds+1
                        count = count+1
                    else:
                        count = count+1
            print("+++++++++++++++++++++")

            if not founds == 0:
                obj_perc = float(founds)/float(count)
                print(str(count)+"/"+str(founds)+"="+str(obj_perc)+"\n")
                founds = 0
                count = 0
                if obj_perc > 0.8:
                    matrix[y_g,x_g] = 1.0
                                    #b_x = box[2]
                                    #b_y = box[3]

            #new average instead of single point, because of clustering
            '''
            avg = 0.0
            count = 0.0
            for b_x in range(box[0],box[2]):
                for b_y in range(box[1],box[3]):
                    if x_grid[x] < b_x and b_x < x_grid[x+1]:
                        if y_grid[y] < b_y and b_y < y_grid[y+1]:
                            avg += 1.0
                    count += 1.0
            #avg = count/avg
            if avg*2.0 > count:
                matrix[y][x] = 1
                b_x = box[2]
                b_y = box[3]
            '''
    #print(matrix)
    return matrix

def set_object(matrix, x_grid, y_grid, bbox):
    for x in range(1,matrix.shape[1]):
        for y in range(1,matrix.shape[0]):
            for i in range(len(bbox)):
                box = bbox[i]
                #mid_x = (box[0]+box[2])/2#(x_grid[x]+x_grid[x+1])/2
                #mid_y = (box[1]+box[3])/2#(y_grid[y]+y_grid[y+1])/2

                #if a point of the bounding box is in the part of the grid it belongs to an object

                for b_x in range(box[0],box[2]):
                    for b_y in range(box[1],box[3]):
                        if x_grid[x] < b_x and b_x < x_grid[x+1]:
                            if y_grid[y] < b_y and b_y < y_grid[y+1]:
                                matrix[y][x] = 1
                                b_x = box[2]
                                b_y = box[3]

                #new average instead of single point, because of clustering
                '''
                avg = 0.0
                count = 0.0

                for b_x in range(box[0],box[2]):
                    for b_y in range(box[1],box[3]):
                        if x_grid[x] < b_x and b_x < x_grid[x+1]:
                            if y_grid[y] < b_y and b_y < y_grid[y+1]:
                                avg += 1.0
                        count += 1.0

                #avg = count/avg

                if avg*2.0 > count:
                    matrix[y][x] = 1
                    b_x = box[2]
                    b_y = box[3]
                '''

    #print(matrix)

    return matrix

def prepare_inputdata(bboxes, images):
    """
    For this GRaz50 Case:
    bboxes = Array of y_coordinates,x_coordinates
    images = label images

    """
    if len(bboxes) == len(images):
        #construct matrix
        all_mat = []

        for i in range(len(bboxes)):
            #Data needed for further work
            img = images[i]
            #boxes = bboxes[i]
            #x_grid = all_grids[i][0]
            #y_grid = all_grids[i][1]
            y_grid,x_grid = bboxes[i]
            #print(np.array(all_grids).shape)
            print("##############################")
            print("IMG SHAPE: "+str(img.shape))
            print(len(x_grid))
            if len(x_grid) <= 0:
                print("ERROR: x_grid empty")
                print(x_grid)
                exit()
            print("MAX XGRID: "+str(x_grid[len(x_grid)-1]))
            print(len(y_grid))
            if len(y_grid) <= 0:
                print("ERROR: y_grid empty")
                print(y_grid)
                exit()
            print("MAX YGRID: "+str(y_grid[len(y_grid)-1]))
            print("##############################")

            #construct empty matrix
            matrix = np.zeros((len(y_grid)-1,len(x_grid)-1))

            #fill matrix with 1 = detected object
            matrix = set_object_alt(img, matrix, x_grid, y_grid)
            update_min_max(matrix)
            print("Max:")
            print(MAX_MAT_X)
            print(MAX_MAT_Y)
            print("Min:")
            print(MIN_MAT_X)
            print(MIN_MAT_Y)
            '''
            HERE WRITE COMPRESSED MATRIX
            '''
            #print(str(i)+" - new compression code")
            #matrix, x_comp_info, y_comp_info = compress_matrix(matrix)
            #write_matrix_comp(i,matrix,x_comp_info,y_comp_info)
            write_matrix(i,matrix)
            write_bboxfile(i,matrix)
            '''
            #
            '''
            grid_img = add_grid_lines( x_grid, y_grid, img)
            label_image = reconstruct_labelimage( x_grid, y_grid, matrix, img)
            cv.imwrite(PATH_RNN_DATASET+RNN_TEST_DIR+"{0:06d}".format(i)+".jpg",grid_img)
            cv.imwrite(PATH_RNN_DATASET+RNN_TEST_DIR+"{0:06d}".format(i)+".png",label_image)
            #cv.imshow('image',img)
            #cv.waitKey(0)
            #cv.destroyAllWindows()

        #remove random object
    else:
        print("prepare_inputdata")
        print("ERROR: Number of Annotations and Images differs.")
        print("bboxes: "+ str(len(bboxes)))
        print("images: "+ str(len(images)))

def compress_sep_matrix(matrix):
    x_len = matrix.shape[1]
    y_len = matrix.shape[0]
    new_matrix = np.copy(matrix)
    x_comp_info = []
    y_comp_info = []
    counter = 0

    for y in range(y_len):
        if not np.any(matrix[y,:] > 0.5):
            new_matrix = np.delete(new_matrix,y-counter,0)
            y_comp_info.append(y)
            counter += 1

    counter = 0

    for x in range(x_len):
        if not np.any(matrix[:,x] > 0.5):
            new_matrix = np.delete(new_matrix,x-counter,1)
            x_comp_info.append(x)
            counter += 1

    return new_matrix, np.array(x_comp_info), np.array(y_comp_info)


def write_sep_matrix_comp(iter,matrix, x_comp_info, y_comp_info):
    if matrix.shape[0] < MAX_SIZE and matrix.shape[1] < MAX_SIZE:
        test_file = "{0:06d}".format(iter)+".txt"
        full_path = PATH_RNN_DATASET+RNN_TEST_DIR+test_file
        comp_full_path = PATH_RNN_DATASET+COMP_INFO_DIR+test_file
        #Write Matrix file
        f = open(full_path, "w")

        for x in range(matrix.shape[0]):
            for y in range(matrix.shape[1]):
                f.write(str(matrix[x][y])+" ")
            f.write("\n")
        f.close()
        #Write Compressing information
        fc = open(comp_full_path, "w")
        fc.write("y ")
        for yc in range(y_comp_info.shape[0]):
            fc.write(str(y_comp_info[yc])+" ")
        fc.write("\nx ")
        for xc in range(x_comp_info.shape[0]):
            fc.write(str(x_comp_info[xc])+" ")
        fc.close()
    else:
        to_delete.append(iter)

def prepare_sep_inputdata(bboxes, images, all_grids):
    if len(bboxes) == len(images) and len(bboxes) == len(all_grids):
        #construct matrix
        all_mat = []

        for i in range(len(bboxes)):
            #Data needed for further work
            img = images[i]
            boxes = bboxes[i]
            x_grid = all_grids[i][0]
            y_grid = all_grids[i][1]
            print(np.array(all_grids).shape)
            print("##############################")
            print("IMG SHAPE: "+str(img.shape))
            print(len(x_grid))
            print("MAX XGRID: "+str(x_grid[len(x_grid)-1]))
            print(len(y_grid))
            print("MAX YGRID: "+str(y_grid[len(y_grid)-1]))
            print("##############################")

            #construct empty matrix
            matrix = np.zeros((len(y_grid)-1,len(x_grid)-1))

            #fill matrix with 1 = detected object
            matrix = set_object(matrix, x_grid, y_grid, boxes)
            update_min_max(matrix)
            print("Max:")
            print(MAX_MAT_X)
            print(MAX_MAT_Y)
            print("Min:")
            print(MIN_MAT_X)
            print(MIN_MAT_Y)
            '''
            HERE WRITE COMPRESSED MATRIX
            '''
            print(str(i)+" - new compression code")
            matrix, x_comp_info, y_comp_info = compress_matrix(matrix)
            write_matrix_comp(i,matrix,x_comp_info,y_comp_info)
            #write_matrix(i,matrix)
            #write_bboxfile(i,matrix)
            '''
            #
            '''
            grid_img = add_grid_lines( x_grid, y_grid, img)
            label_image = reconstruct_labelimage( x_grid, y_grid, matrix, img)
            cv.imwrite(PATH_RNN_DATASET+RNN_TEST_DIR+"{0:06d}".format(i)+".jpg",grid_img)
            cv.imwrite(PATH_RNN_DATASET+RNN_TEST_DIR+"{0:06d}".format(i)+".png",label_image)
            #cv.imshow('image',img)
            #cv.waitKey(0)
            #cv.destroyAllWindows()

        #remove random object
    else:
        print("prepare_inputdata")
        print("ERROR: Number of Annotations and Images differs.")
        print("bboxes: "+ str(len(bboxes)))
        print("images: "+ str(len(images)))

def prepare_gtdata(bboxes, images):
    if len(bboxes) == len(images):
        pass
    else:
        print("prepare_gtdata")
        print("ERROR: Number of Annotations and Images differs.")
        print("bboxes: "+ str(len(bboxes)))
        print("images: "+ str(len(images)))




def write_data(file_name):
    pass

def delete_toobig():
    for i in range(len(to_delete)):
        del_pos = to_delete[i]
        file_name = PATH_RNN_DATASET+RNN_TEST_DIR+"{0:06d}".format(del_pos)
        os.remove(file_name+".jpg")
        os.remove(file_name+".png")
        #os.remove(file_name+".txt")
        print("File: "+str(del_pos)+" deleted!")

def cluster_gridlines(bboxen):
    for i in range(len(bboxen)):
        for j in range(len(bboxen)):
            if not i == j:
                for b in range(4):
                    dif = abs(bboxen[i][b]-bboxen[j][b])
                    if dif < MAX_THRESHOLD:
                        new_value = float(bboxen[i][b]+bboxen[j][b])/2
                        bboxen[i][b] = int(new_value)
                        bboxen[j][b] = int(new_value)
    return bboxen

def cluster_coordinates(y_coordinates, x_coordinates):
    y_new = []
    x_new = []
    for y in range(len(y_coordinates)-1):
        dif = abs(y_coordinates[y]-y_coordinates[y+1])
        if dif < MAX_THRESHOLD:
            new_value = float(y_coordinates[y]+y_coordinates[y+1])/2.0
            y_new.append(int(new_value))
            y = y+1
        else:
            y_new.append(y_coordinates[y])
    y_new.append(y_coordinates[len(y_coordinates)-1])

    for x in range(len(x_coordinates)-1):
        dif = abs(x_coordinates[x]-x_coordinates[x+1])
        if dif < MAX_THRESHOLD:
            new_value = float(x_coordinates[x]+x_coordinates[x+1])/2.0
            x_new.append(int(new_value))
            x = x+1
        else:
            x_new.append(x_coordinates[x])
    x_new.append(x_coordinates[len(x_coordinates)-1])

    return y_new, x_new

def write_bboxfile(iter, matrix):
    """
    if matrix.shape[0] < MAX_SIZE and matrix.shape[1] < MAX_SIZE:
        test_file = "{0:06d}".format(iter)+".txt"
        full_path = PATH_RNN_DATASET+RNN_TEST_DIR+test_file

        f = open(full_path, "w")

        for x in range(matrix.shape[0]):
            for y in range(matrix.shape[1]):
                f.write(str(matrix[x][y])+" ")
            f.write("\n")

        f.close()
    else:
        to_delete.append(iter)
    """
    path = PATH_RNN_DATASET+RNN_TEST_DIR+"c{0:06d}".format(iter)+".txt"
    f = open(path, "w")

    for j in range(matrix.shape[0]):
        for i in range(matrix.shape[1]):
            if matrix[j][i] == 1.0:
                f.write("{0:03d} {1:03d}\n".format(j,i))
    f.close()






def main():
    print(PATH_BASEDATASET+IMAGESETS_SEG_DIR)
    #LOAD ALL XML FILES
    #all_bboxes = []
    #all_xml = list_xml(PATH_BASEDATASET+IMAGESETS_SEG_DIR)
    #print(len(all_xml))
    #for i in range(len(all_xml)):
        #bboxes = parse_xml(all_xml[i])
        #all_bboxes.append(cluster_gridlines(bboxes))
        #print("Objects: "+str(len(bboxes)))
    #LOAD ALL IMAGES
    all_images = list_imgs(PATH_BASEDATASET+IMAGESETS_SEG_DIR)
    images_loaded = []
    all_coordinates = []
    for i in range(len(all_images)):
        tmp_img = cv.imread(all_images[i])
        #y_coordinate, x_coordinate = image_to_grid(tmp_img)
        #all_coordinates.append(cluster_coordinates(y_coordinate, x_coordinate))#TODO cluster_coordinates makes an error
        all_coordinates.append(image_to_grid(tmp_img))
        print("#######################")
        #print(all_xml[i])
        print(all_images[i])
        print("#######################")
        images_loaded.append(tmp_img.copy())

    #CONSTRUCT GRIDS
    #all_grids = extract_grids(all_bboxes, images_loaded)
    #new_all_grids = []

    #CLUSTER GRIDS
    #for i in range(len(all_grids)):
        #new_all_grids.append(cluster_coordinates(all_grids[i]))


    #GENERATE TRAININGSDATA USING GROUNDTRUTH
    print("all_grids: "+str(np.array(all_coordinates).shape))
    #print("new_all_grids: "+str(np.array(new_all_grids).shape))
    #prepared_td = prepare_inputdata(all_bboxes, images_loaded, all_grids)
    prepared_td = prepare_inputdata(all_coordinates, images_loaded)
    #prepared_gt = prepare_gtdata(bboxes, images_loaded, all_grids)

    delete_toobig()


if __name__ == '__main__':
    main()
