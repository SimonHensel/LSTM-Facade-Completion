def random_erase(matrix_list, number_of_objects, coordinates_array=None):
    new_matrix_list = []
    if coordinate_array == None:
        for x in range(len(matrix_list)):
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
    else:
        for c in range(0,len(coordinate_list),2):
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
