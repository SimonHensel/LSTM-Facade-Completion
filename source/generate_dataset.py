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

PATH_BASEDATASET = "/media/DATA/simon/alt_branch/tf-faster-rcnn/data/VOCdevkit/VOC2007_backup2/"
ANNOTATION_DIR = "Annotations/"
IMAGE_DIR = "JPEGImages/"
IMAGESETS_DIR = "ImageSets/"
IMAGESETS_LAYOUT_DIR = "Layout/"
IMAGESETS_MAIN_DIR = "Main/"
IMAGESETS_SEG_DIR = "Segmentation/"
DEPTH_DIR = "DepthMaps/"

PATH_RNN_DATASET = "/media/DATA/simon/mdlstm_modified/tensorflow-multi-dimensional-lstm/data/"
RNN_TEST_DIR = "tests_cluster_compressed/"
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
            grid_img[y][x_grid[x]] = [0,0,255]

    for y in range(len(y_grid)):
        for x in range(grid_img.shape[1]):
            #print(y_grid[y])
            grid_img[y_grid[y]][x] = [255,0,0]

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
        if (current_file_abs_path.endswith(".jpg")):
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
                cv.rectangle(label_image,(x_grid[colums],y_grid[rows]),(x_grid[colums+1],y_grid[rows+1]),(0,0,255),3)

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

def prepare_inputdata(bboxes, images, all_grids):
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
    print(PATH_BASEDATASET+ANNOTATION_DIR)
    #LOAD ALL XML FILES
    all_bboxes = []
    all_xml = list_xml(PATH_BASEDATASET+ANNOTATION_DIR)
    #print(len(all_xml))
    for i in range(len(all_xml)):
        bboxes = parse_xml(all_xml[i])
        all_bboxes.append(cluster_gridlines(bboxes))
        #print("Objects: "+str(len(bboxes)))
    #LOAD ALL IMAGES
    all_images = list_imgs(PATH_BASEDATASET+IMAGE_DIR)
    images_loaded = []
    for i in range(len(all_images)):
        print("#######################")
        print(all_xml[i])
        print(all_images[i])
        print("#######################")
        tmp_img = cv.imread(all_images[i])
        images_loaded.append(tmp_img.copy())

    #CONSTRUCT GRIDS
    all_grids = extract_grids(all_bboxes, images_loaded)
    new_all_grids = []

    #CLUSTER GRIDS
    for i in range(len(all_grids)):
        new_all_grids.append(cluster_gridlines(all_grids[i]))


    #GENERATE TRAININGSDATA USING GROUNDTRUTH
    print("all_grids: "+str(np.array(all_grids).shape))
    print("new_all_grids: "+str(np.array(new_all_grids).shape))
    #prepared_td = prepare_inputdata(all_bboxes, images_loaded, all_grids)
    prepared_td = prepare_inputdata(all_bboxes, images_loaded, new_all_grids)
    #prepared_gt = prepare_gtdata(bboxes, images_loaded, all_grids)

    delete_toobig()


if __name__ == '__main__':
    main()
