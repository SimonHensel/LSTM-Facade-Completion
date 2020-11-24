#from scipy.linalg import hilbert
from hilbertcurve.hilbertcurve import HilbertCurve
import numpy as np
import tensorflow as tf

MATRIX_SIZE = 16

def translate_coordinates(x, y, hilbert_curve):
    dist = hilbert_curve.distance_from_coordinates([x,y])
    return dist
'''
def to_hilbert(numpy_array, b, h, w, c):
    y_size = h#numpy_array.shape[0]
    x_size = w#numpy_array.shape[1]
    print("###to_hilbert:###")
    print(y_size)
    print(x_size)
    print(b)
    print(c)
    print("###start loop###")

    hilbert_curve = HilbertCurve(x_size,2)
    hilbert_array = []#tf.zeros((b, w * h, c))


    for i in range(b):
        for y in range(y_size):
            for x in range(x_size):
                h_c = translate_coordinates(x,y,hilbert_curve)
                print(str(i)+" "+str(h_c)+" "+str(c)+" | "+str(y)+" "+str(x))
                hilbert_array[i,h_c,c].assign(numpy_array[y,x])

    return hilbert_array
'''

def to_hilbert(tensor_array, b, h, w, c):
    y_size = h#numpy_array.shape[0]
    x_size = w#numpy_array.shape[1]
    print("###to_hilbert:###")
    print(y_size)
    print(x_size)
    print(b)
    print(c)
    print("###start loop###")

    hilbert_curve = HilbertCurve(x_size,2)
    hilbert_array = []#tf.zeros((b, w * h, c))
    hilbert_batch = []
    #IDEE erst Reihenfolge bestimmen, dann in stack packen
    hilbert_ref = []
    x_ref = []
    y_ref = []
    for y in range(y_size):
        for x in range(x_size):
            hilbert_ref.append(translate_coordinates(y,x,hilbert_curve))
            x_ref.append(x)
            y_ref.append(y)

    _ , x_ref = zip(*sorted(zip(hilbert_ref, x_ref)))
    _ , y_ref = zip(*sorted(zip(hilbert_ref, y_ref)))


    batch_list = tf.unstack(tensor_array)
    print(batch_list)
    for i in range(b):
        print(i)
        batch = batch_list[i] # batch = []
        hilbert_batch = []
        for d in range(w * h):
            hilbert_batch.append(tensor_array[i,y_ref[d],x_ref[d]])
        hilbert_array.append(hilbert_batch)

    result = tf.stack(hilbert_array)

    return result


def test():
    numpy_array = np.arange(MATRIX_SIZE*MATRIX_SIZE)
    print("##########################")
    print(numpy_array)
    numpy_array = np.reshape(numpy_array,(MATRIX_SIZE,MATRIX_SIZE))

    new_array = to_hilbert(numpy_array)

    print("##########################")
    print(numpy_array)
    print("##########################")
    print(new_array)

if __name__ == "__main__":
    print("Testing Hilber Matrix to Hilbert Array...")
    test()
