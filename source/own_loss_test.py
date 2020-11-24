from time import time
import os
import math
import cv2 as cv

import argparse
import logging
import numpy as np
import tensorflow as tf
from enum import Enum

SIZE = 64 #3 10 64
BATCH_SIZE = 16
EPSILON = 5
'''
SHIFT_TEST = tf.constant(
[[1.0,2.0,3.0],
[4.0, 5.0, 6.0],
[7.0, 8.0, 9.0]]
)
'''
SHIFT_TEST = tf.constant(
[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
[11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
[21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0],
[31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0],
[41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0],
[51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0],
[61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0],
[71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0],
[81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0],
[91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 100.0]])

NONZERO_TEST = tf.constant(
[[0.0001, 0.0001, 0.0001],
[0.0001, 1.0, 0.0001],
[0.0001, 0.0001, 0.0001]]
)

#ONE_TENSOR = tf.constant()
'''
Y_TENSOR = tf.constant(
[[0.0001, 0.0001, 0.0001],
[0.0001, 1.0, 0.0001],
[0.0001, 0.0001, 0.0001]]
)

X_SHAPE_WRONG_TENSOR = tf.constant(
[[0.2001, 0.3001, 0.2001],
[0.3001, 0.9000, 0.3001],
[0.2001, 0.2001, 0.2001]]
)

X_SHAPE_RIGHT_TENSOR = tf.constant(
[[0.1001, 0.1001, 0.1001],
[0.1001, 0.8000, 0.1001],
[0.1001, 0.1001, 0.1001]]
)
'''

#MORE COMPLEX TENSORS
Y_TENSOR = tf.constant(
[[0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001],
[0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001],
[0.0001, 0.0001, 1.0000, 0.0001, 1.0000, 0.0001, 1.0000, 0.0001, 1.0000, 0.0001],
[0.0001, 0.0001, 1.0000, 0.0001, 1.0000, 0.0001, 1.0000, 0.0001, 1.0000, 0.0001],
[0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001],
[0.0001, 0.0001, 1.0000, 0.0001, 1.0000, 0.0001, 0.0001, 0.0001, 1.0000, 0.0001],
[0.0001, 0.0001, 1.0000, 0.0001, 1.0000, 0.0001, 1.0000, 0.0001, 1.0000, 0.0001],
[0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 1.0000, 0.0001, 0.0001, 0.0001],
[0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 1.0000, 0.0001, 0.0001, 0.0001],
[0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 1.0000, 0.0001, 0.0001, 0.0001]])#, value_index=(10, 10),dtype=tf.float32)

X_SHAPE_WRONG_TENSOR = tf.constant(
[[0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001],
[0.0001, 0.2001, 0.3001, 0.2001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001],
[0.0001, 0.3001, 0.9000, 0.3001, 1.0000, 0.0001, 1.0000, 0.0001, 1.0000, 0.0001],
[0.0001, 0.4001, 0.9000, 0.4001, 1.0000, 0.0001, 1.0000, 0.0001, 1.0000, 0.0001],
[0.0001, 0.2001, 0.2001, 0.2001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001],
[0.0001, 0.0001, 1.0000, 0.0001, 1.0000, 0.0001, 0.0001, 0.0001, 1.0000, 0.0001],
[0.0001, 0.0001, 1.0000, 0.0001, 1.0000, 0.0001, 1.0000, 0.0001, 1.0000, 0.0001],
[0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 1.0000, 0.0001, 0.0001, 0.0001],
[0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 1.0000, 0.0001, 0.0001, 0.0001],
[0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 1.0000, 0.0001, 0.0001, 0.0001]])#, value_index=(10, 10), dtype=tf.float32)

X_SHAPE_RIGHT_TENSOR = tf.constant(
[[0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001],
[0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001],
[0.0001, 0.1001, 0.8000, 0.1001, 1.0000, 0.0001, 1.0000, 0.0001, 1.0000, 0.0001],
[0.0001, 0.1001, 0.8000, 0.1001, 1.0000, 0.0001, 1.0000, 0.0001, 1.0000, 0.0001],
[0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001],
[0.0001, 0.0001, 1.0000, 0.0001, 1.0000, 0.0001, 0.0001, 0.0001, 1.0000, 0.0001],
[0.0001, 0.0001, 1.0000, 0.0001, 1.0000, 0.0001, 1.0000, 0.0001, 1.0000, 0.0001],
[0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 1.0000, 0.0001, 0.0001, 0.0001],
[0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 1.0000, 0.0001, 0.0001, 0.0001],
[0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 1.0000, 0.0001, 0.0001, 0.0001]])#, value_index=(10, 10), dtype=tf.float32)


'''
Y = [[0.0001, 0.0001, 0.0001],
[0.0001, 1.0, 0.0001],
[0.0001, 0.0001, 0.0001]]


X_SHAPE_WRONG = [[0.2001, 0.3001, 0.2001],
[0.3001, 0.9000, 0.3001],
[0.2001, 0.2001, 0.2001]]


X_SHAPE_RIGHT = [[0.1001, 0.1001, 0.1001],
[0.1001, 0.8000, 0.1001],
[0.1001, 0.1001, 0.1001]]
'''


Y =[[0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001],
[0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001],
[0.0001, 0.0001, 1.0000, 0.0001, 1.0000, 0.0001, 1.0000, 0.0001, 1.0000, 0.0001],
[0.0001, 0.0001,1.0000, 0.0001, 1.0000, 0.0001, 1.0000, 0.0001, 1.0000, 0.0001],
[0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001],
[0.0001, 0.0001, 1.0000, 0.0001, 1.0000, 0.0001, 0.0001, 0.0001, 1.0000, 0.0001],
[0.0001, 0.0001, 1.0000, 0.0001, 1.0000, 0.0001, 1.0000, 0.0001, 1.0000, 0.0001],
[0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 1.0000, 0.0001, 0.0001, 0.0001],
[0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 1.0000, 0.0001, 0.0001, 0.0001],
[0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 1.0000, 0.0001, 0.0001, 0.0001]]#, value_index=(10, 10),dtype=tf.float32)

X_SHAPE_WRONG =[[0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001],
[0.0001, 0.2001, 0.3001, 0.2001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001],
[0.0001, 0.3001, 0.9000, 0.3001, 1.0000, 0.0001, 1.0000, 0.0001, 1.0000, 0.0001],
[0.0001, 0.4001, 0.9000, 0.4001, 1.0000, 0.0001, 1.0000, 0.0001, 1.0000, 0.0001],
[0.0001, 0.2001, 0.2001, 0.2001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001],
[0.0001, 0.0001, 1.0000, 0.0001, 1.0000, 0.0001, 0.0001, 0.0001, 1.0000, 0.0001],
[0.0001, 0.0001, 1.0000, 0.0001, 1.0000, 0.0001, 1.0000, 0.0001, 1.0000, 0.0001],
[0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 1.0000, 0.0001, 0.0001, 0.0001],
[0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 1.0000, 0.0001, 0.0001, 0.0001],
[0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 1.0000, 0.0001, 0.0001, 0.0001]]#, value_index=(10, 10), dtype=tf.float32)

X_SHAPE_RIGHT =[[0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001],
[0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001],
[0.0001, 0.1001, 0.8000, 0.1001, 1.0000, 0.0001, 1.0000, 0.0001, 1.0000, 0.0001],
[0.0001, 0.1001, 0.8000, 0.1001, 1.0000, 0.0001, 1.0000, 0.0001, 1.0000, 0.0001],
[0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001],
[0.0001, 0.0001, 1.0000, 0.0001, 1.0000, 0.0001, 0.0001, 0.0001, 1.0000, 0.0001],
[0.0001, 0.0001, 1.0000, 0.0001, 1.0000, 0.0001, 1.0000, 0.0001, 1.0000, 0.0001],
[0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 1.0000, 0.0001, 0.0001, 0.0001],
[0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 1.0000, 0.0001, 0.0001, 0.0001],
[0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 1.0000, 0.0001, 0.0001, 0.0001]]#, value_index=(10, 10), dtype=tf.float32)

"""
  X_MOD = tf.Tensor(
  [[0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001]
  [0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001]
  [0.0001 0.0001 1.0000 0.0001 1.0000 0.0001 1.0000 0.0001 1.0000 0.0001]
  [0.0001 0.0001 1.0000 0.0001 1.0000 0.0001 1.0000 0.0001 1.0000 0.0001]
  [0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001]
  [0.0001 0.0001 1.0000 0.0001 1.0000 0.0001 0.0001 0.0001 1.0000 0.0001]
  [0.0001 0.0001 1.0000 0.0001 1.0000 0.0001 1.0000 0.0001 1.0000 0.0001]
  [0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 1.0000 0.0001 0.0001 0.0001]
  [0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 1.0000 0.0001 0.0001 0.0001]
  [0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 1.0000 0.0001 0.0001 0.0001]], shape=(10, 10), dtype=float32)

   X_RISK = tf.Tensor(
   [[0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001]
   [0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001]
   [0.0001 0.0001 1.0000 0.0001 1.0000 0.0001 1.0000 0.0001 1.0000 0.0001]
   [0.0001 0.0001 1.0000 0.0001 1.0000 0.0001 1.0000 0.0001 1.0000 0.0001]
   [0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001]
   [0.0001 0.0001 1.0000 0.0001 1.0000 0.0001 0.0001 0.0001 1.0000 0.0001]
   [0.0001 0.0001 1.0000 0.0001 1.0000 0.0001 1.0000 0.0001 1.0000 0.0001]
   [0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 1.0000 0.0001 0.0001 0.0001]
   [0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 1.0000 0.0001 0.0001 0.0001]
   [0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 1.0000 0.0001 0.0001 0.0001]], shape=(10, 10), dtype=float32)

"""

def get_distance(array, DistType):
    print(array)
    array_np = np.array(array)*255
    array_np = np.expand_dims(array_np.astype(np.uint8) ,axis=-1)
    print(array_np.shape)
    print(array_np)
    if DistType == 1:
        result = cv.distanceTransform(cv.bitwise_not(array_np), distanceType=cv.DIST_L1,maskSize=3)
    elif DistType == 2:
        result = cv.distanceTransform(cv.bitwise_not(array_np), distanceType=cv.DIST_L2,maskSize=5)
    else:
        print("ERROR - wrong DISTTYPE")
        result = None

    return result

def shape_loss(x,y,n_size,m_size):
    """
    This is the shape loss
    If the objects have clear boarders, the loss gets low
    If the boarders arent clear the loss gets high
    """
    def f(x):
        if x < 0.1:
            #print("x < 0.1")
            result = (-x+0.1)*(-x+0.1)*(-x+0.1)
        elif x > 0.9:
            #print("x > 0.9")
            result = (x-0.9)*(x-0.9)*(x-0.9)
        else:
            #print("0.1 < x < 0.9")
            result = (-5.6*((x-0.5)*(x-0.5)))+1

        #print(str(x)+" -> "+str(result))

        return result

    print("Calc shape loss...")
    loss = 0.0
    for n in range(1, n_size-1):
        for m in range(1, m_size-1):
            if y[n][m] == 1.0:
                tmp1 = f(abs(x[n][m]-x[n+1][m]))
                #print("1. "+str(tmp1))
                loss += tmp1
                tmp2 = f(abs(x[n][m]-x[n-1][m]))
                #print("2. "+str(tmp2))
                loss += tmp2
                tmp3 = f(abs(x[n][m]-x[n][m+1]))
                #print("3. "+str(tmp3))
                loss += tmp3
                tmp4 = f(abs(x[n][m]-x[n][m-1]))
                #print("4. "+str(tmp4))
                loss += tmp4
                #print(loss)

    return loss




def own_tensor_shift(in_tensor, t_size, direction):
    """
    in_tensor: input tensor
    t_size: size
    direction: integer. 1 up, 2 right, 3 down, 4 left
    """
    in_tensor = tf.squeeze(in_tensor)
    #print(in_tensor)
    if direction == 1:
        s_shift = 1
        T_shift = tf.zeros((t_size, t_size), tf.float32)
        tmp = []
        #print("for start:")
        for i in range(t_size):
            T_concat = tf.concat([in_tensor[ s_shift:, i],T_shift[i, :s_shift]], axis = 0)
            #print(sess.run(T_concat))
            tmp.append(T_concat)
        T_shift = tf.stack(tmp, axis=1)
    elif direction == 2:
        s_shift = 1
        T_shift = tf.zeros((t_size, t_size), tf.float32)
        tmp = []
        for i in range(t_size):
            tmp.append(tf.concat([T_shift[i, :s_shift],in_tensor[i, :t_size - s_shift]], axis = 0))
        T_shift = tf.stack(tmp)
    elif direction == 3:
        s_shift = 1
        T_shift = tf.zeros((t_size, t_size), tf.float32)
        tmp = []
        for i in range(t_size):
            T_concat = tf.concat([T_shift[i, :s_shift] ,in_tensor[ s_shift:, i]], axis = 0)
            #print(sess.run(T_concat))
            tmp.append(T_concat)
        T_shift = tf.stack(tmp, axis=1)
    elif direction == 4:
        s_shift = 1
        T_shift = tf.zeros((t_size, t_size), tf.float32)
        tmp = []
        for i in range(t_size):
            tmp.append(tf.concat([in_tensor[i, s_shift:],T_shift[i, :s_shift]], axis = 0))
        T_shift = tf.stack(tmp)
    #print(" ")

    return T_shift


def shape_loss_tensor(x_tensor,y_tensor, debug=False):
    """
    x = tf.constant([1, 4])
    y = tf.constant([2, 5])
    z = tf.constant([3, 6])
    tf.stack([x, y, z])  # [[1, 4], [2, 5], [3, 6]] (Pack along first dim.)
    tf.stack([x, y, z], axis=1)  # [[1, 2, 3], [4, 5, 6]]
    ############################################################################
    t1 = [[1, 2, 3], [4, 5, 6]]
    t2 = [[7, 8, 9], [10, 11, 12]]
    tf.concat([t1, t2], 0)  # [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    tf.concat([t1, t2], 1)  # [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]

    # tensor t3 with shape [2, 3]
    # tensor t4 with shape [2, 3]
    tf.shape(tf.concat([t3, t4], 0))  # [4, 3]
    tf.shape(tf.concat([t3, t4], 1))  # [2, 6]
    """
    """
    This is the shape loss
    If the objects have clear boarders, the loss gets low
    If the boarders arent clear the loss gets high
    """
    x_tensor = tf.squeeze(x_tensor)
    y_tensor = tf.squeeze(y_tensor)
    if debug:
        print("START")
        print(x_tensor)
        print(sess.run(x_tensor))
        print(y_tensor)
        print(sess.run(y_tensor))
    #SKIPPING ALL ENTRIES WHICH ARE NOT 1.0
    #x_skip =  tf.where(tf.equal(y_tensor,1.0), x_tensor, 0)
    '''
    x_skip =  tf.where(tf.equal(y_tensor,1.0), x_tensor, tf.zeros_like(x_tensor))
    if debug:
        print("SKIPPING ALL ENTRIES WHICH ARE NOT 1.0")
        print(x_skip)
        print(sess.run(x_skip))
    '''

    #CALCULATE ABSOLUTE DIFFERENCE BETWEEN NEIGHBORS AT THE 1.0 ENTRIES
    #T_shift = tf.zeros_like(x_skip)
    #tmp = []
    #tf.concat([T_shift[ :S[i, 0]],T[i, :17 - S[i,0]]], axis = 0)
    #T_shift = tf.stack(tmp)

    #abs1 = tf.abs(tf.subtract(x_skip, tf.roll(x_skip, shift=1, axis=0)))
    #abs2 = tf.abs(tf.subtract(x_skip, tf.roll(x_skip, shift=1, axis=1)))
    #abs3 = tf.abs(tf.subtract(x_skip, tf.roll(x_skip, shift=-1, axis=0)))
    #abs4 = tf.abs(tf.subtract(x_skip, tf.roll(x_skip, shift=-1, axis=1)))

    abs1 = tf.abs(tf.subtract(x_tensor, own_tensor_shift(x_tensor, SIZE, 1)))
    abs2 = tf.abs(tf.subtract(x_tensor, own_tensor_shift(x_tensor, SIZE, 2)))
    abs3 = tf.abs(tf.subtract(x_tensor, own_tensor_shift(x_tensor, SIZE, 3)))
    abs4 = tf.abs(tf.subtract(x_tensor, own_tensor_shift(x_tensor, SIZE, 4)))
    if debug:
        print("CALCULATE ABSOLUTE DIFFERENCE BETWEEN NEIGHBORS AT THE 1.0 ENTRIES")
        print(abs1)
        print(sess.run(abs1))
        print(abs2)
        print(sess.run(abs2))
        print(abs3)
        print(sess.run(abs3))
        print(abs4)
        print(sess.run(abs4))

    abs1 =  tf.where(tf.equal(y_tensor,1.0), abs1, tf.zeros_like(abs1))
    abs2 =  tf.where(tf.equal(y_tensor,1.0), abs2, tf.zeros_like(abs2))
    abs3 =  tf.where(tf.equal(y_tensor,1.0), abs3, tf.zeros_like(abs3))
    abs4 =  tf.where(tf.equal(y_tensor,1.0), abs4, tf.zeros_like(abs4))

    if debug:
        print("SKIPPING ALL ENTRIES WHICH ARE NOT 1.0")
        print(abs1)
        print(sess.run(abs1))
        print(abs2)
        print(sess.run(abs2))
        print(abs3)
        print(sess.run(abs3))
        print(abs4)
        print(sess.run(abs4))

    #APPLY FUNCTION F
    abs1_low = tf.boolean_mask(abs1, tf.less(abs1, 0.1))
    abs1_middle = tf.boolean_mask(abs1, tf.logical_and(tf.less(abs1, 0.9),tf.greater(abs1, 0.1)))
    abs1_high = tf.boolean_mask(abs1, tf.greater(abs1, 0.9))

    abs2_low = tf.boolean_mask(abs2, tf.less(abs2, 0.1))
    abs2_middle = tf.boolean_mask(abs2, tf.logical_and(tf.less(abs2, 0.9),tf.greater(abs2, 0.1)))
    abs2_high = tf.boolean_mask(abs2, tf.greater(abs2, 0.9))

    abs3_low = tf.boolean_mask(abs3, tf.less(abs3, 0.1))
    abs3_middle = tf.boolean_mask(abs3, tf.logical_and(tf.less(abs3, 0.9),tf.greater(abs3, 0.1)))
    abs3_high = tf.boolean_mask(abs3, tf.greater(abs3, 0.9))

    abs4_low = tf.boolean_mask(abs4, tf.less(abs4, 0.1))
    abs4_middle = tf.boolean_mask(abs4, tf.logical_and(tf.less(abs4, 0.9),tf.greater(abs4, 0.1)))
    abs4_high = tf.boolean_mask(abs4, tf.greater(abs4, 0.9))

    if debug:
        print("APPLY FUNCTION F")
        print(abs1_low)
        print(sess.run(abs1_low))
        print(abs1_middle)
        print(sess.run(abs1_middle))
        print(abs1_high)
        print(sess.run(abs1_high))
        print("----------------")
        print(abs2_low)
        print(sess.run(abs2_low))
        print(abs2_middle)
        print(sess.run(abs2_middle))
        print(abs2_high)
        print(sess.run(abs2_high))
        print("----------------")
        print(abs3_low)
        print(sess.run(abs3_low))
        print(abs3_middle)
        print(sess.run(abs3_middle))
        print(abs3_high)
        print(sess.run(abs3_high))
        print("----------------")
        print(abs4_low)
        print(sess.run(abs4_low))
        print(abs4_middle)
        print(sess.run(abs4_middle))
        print(abs4_high)
        print(sess.run(abs4_high))
        print("----------------")

    #(-x + 0.1)³
    abs1_low = tf.multiply(abs1_low,-1.0)
    abs1_low = tf.add(abs1_low, 0.1)
    abs1_low = tf.multiply(tf.multiply(abs1_low,abs1_low),abs1_low)

    abs2_low = tf.multiply(abs2_low,-1.0)
    abs2_low = tf.add(abs2_low, 0.1)
    abs2_low = tf.multiply(tf.multiply(abs2_low,abs2_low),abs2_low)

    abs3_low = tf.multiply(abs3_low,-1.0)
    abs3_low = tf.add(abs3_low, 0.1)
    abs3_low = tf.multiply(tf.multiply(abs3_low,abs3_low),abs3_low)

    abs4_low = tf.multiply(abs4_low,-1.0)
    abs4_low = tf.add(abs4_low, 0.1)
    abs4_low = tf.multiply(tf.multiply(abs4_low,abs4_low),abs4_low)

    if debug:
        print("(-x + 0.1)³")
        print(abs1_low)
        print(sess.run(abs1_low))
        print(abs2_low)
        print(sess.run(abs2_low))
        print(abs3_low)
        print(sess.run(abs3_low))
        print(abs4_low)
        print(sess.run(abs4_low))


    #-6.5(x-0.5)²+1

    abs1_middle = tf.subtract(abs1_middle, 0.5)
    abs1_middle = tf.multiply(abs1_middle,abs1_middle)
    abs1_middle = tf.multiply(abs1_middle,-6.5)
    abs1_middle = tf.add(abs1_middle, 1.0)

    abs2_middle = tf.subtract(abs2_middle, 0.5)
    abs2_middle = tf.multiply(abs2_middle,abs2_middle)
    abs2_middle = tf.multiply(abs2_middle,-6.5)
    abs2_middle = tf.add(abs2_middle, 1.0)

    abs3_middle = tf.subtract(abs3_middle, 0.5)
    abs3_middle = tf.multiply(abs3_middle,abs3_middle)
    abs3_middle = tf.multiply(abs3_middle,-6.5)
    abs3_middle = tf.add(abs3_middle, 1.0)

    abs4_middle = tf.subtract(abs4_middle, 0.5)
    abs4_middle = tf.multiply(abs4_middle,abs4_middle)
    abs4_middle = tf.multiply(abs4_middle,-6.5)
    abs4_middle = tf.add(abs4_middle, 1.0)

    if debug:
        print("-6.5(x-0.5)²+1")
        print(abs1_middle)
        print(sess.run(abs1_middle))
        print(abs2_middle)
        print(sess.run(abs2_middle))
        print(abs3_middle)
        print(sess.run(abs3_middle))
        print(abs4_middle)
        print(sess.run(abs4_middle))

    #(x-0.9)³

    abs1_high = tf.subtract(abs1_high,0.9)
    abs1_high = tf.multiply(tf.multiply(abs1_high,abs1_high),abs1_high)

    abs2_high = tf.subtract(abs2_high,0.9)
    abs2_high = tf.multiply(tf.multiply(abs2_high,abs2_high),abs2_high)

    abs3_high = tf.subtract(abs3_high,0.9)
    abs3_high = tf.multiply(tf.multiply(abs3_high,abs3_high),abs3_high)

    abs4_high = tf.subtract(abs4_high,0.9)
    abs4_high = tf.multiply(tf.multiply(abs4_high,abs4_high),abs4_high)

    if debug:
        print("(x-0.9)³")
        print(abs1_high)
        print(sess.run(abs1_high))
        print(abs2_high)
        print(sess.run(abs2_high))
        print(abs3_high)
        print(sess.run(abs3_high))
        print(abs4_high)
        print(sess.run(abs4_high))

    #ADD TOGETHER FOR FINAL LOSS
    loss1_low = tf.reduce_sum(abs1_low)
    loss1_middle = tf.reduce_sum(abs1_middle)
    loss1_high = tf.reduce_sum(abs1_high)

    loss2_low = tf.reduce_sum(abs2_low)
    loss2_middle = tf.reduce_sum(abs2_middle)
    loss2_high = tf.reduce_sum(abs2_high)

    loss3_low = tf.reduce_sum(abs3_low)
    loss3_middle = tf.reduce_sum(abs3_middle)
    loss3_high = tf.reduce_sum(abs3_high)

    loss4_low = tf.reduce_sum(abs4_low)
    loss4_middle = tf.reduce_sum(abs4_middle)
    loss4_high = tf.reduce_sum(abs4_high)

    loss = tf.add(loss1_low, loss1_middle)
    loss = tf.add(loss, loss1_high)
    loss = tf.add(loss, loss2_low)
    loss = tf.add(loss, loss2_middle)
    loss = tf.add(loss, loss2_high)
    loss = tf.add(loss, loss3_low)
    loss = tf.add(loss, loss3_middle)
    loss = tf.add(loss, loss3_high)
    loss = tf.add(loss, loss4_low)
    loss = tf.add(loss, loss4_middle)
    loss = tf.add(loss, loss4_high)

    if debug:
        print("ADD TOGETHER FOR FINAL LOSS")
        print(abs1_low)
        print(sess.run(abs1_low))
        print(abs1_middle)
        print(sess.run(abs1_middle))
        print(abs1_high)
        print(sess.run(abs1_high))
        print("----------------")
        print(abs2_low)
        print(sess.run(abs2_low))
        print(abs2_middle)
        print(sess.run(abs2_middle))
        print(abs2_high)
        print(sess.run(abs2_high))
        print("----------------")
        print(abs3_low)
        print(sess.run(abs3_low))
        print(abs3_middle)
        print(sess.run(abs3_middle))
        print(abs3_high)
        print(sess.run(abs3_high))
        print("----------------")
        print(abs4_low)
        print(sess.run(abs4_low))
        print(abs4_middle)
        print(sess.run(abs4_middle))
        print(abs4_high)
        print(sess.run(abs4_high))
        print("----------------")
        print("FINAL LOSS")
        print(loss)
    num_loss = loss
    if debug:
        print(num_loss)

    return num_loss

def distbce_loss_batch(x_batches, y_batches, y_dist1_batches, y_dist2_batches, logits, debug=False):
    lloss = tf.losses.log_loss(y_batches,x_batches)
    dloss = binary_cross_entropy_with_distance(
        tf.squeeze(x_batches),
        tf.squeeze(y_batches),
        tf.squeeze(y_dist1_batches),
        tf.squeeze(y_dist2_batches),
        tf.squeeze(logits))
    '''
    dloss = binary_cross_entropy_with_distance(
        tf.squeeze(x_batches[0]),
        tf.squeeze(y_batches[0]),
        tf.squeeze(y_dist1_batches[0]),
        tf.squeeze(y_dist2_batches[0]),
        tf.squeeze(logits[0]))

    for b in range(1,BATCH_SIZE):
        dloss = tf.add(binary_cross_entropy_with_distance(
            tf.squeeze(x_batches[b]),
            tf.squeeze(y_batches[b]),
            tf.squeeze(y_dist1_batches[b]),
            tf.squeeze(y_dist2_batches[b]),
            tf.squeeze(logits[b])),dloss)
    print(dloss)
    '''
    dloss = tf.divide(dloss, float(BATCH_SIZE))
    loss = tf.add(lloss, dloss)
    loss = tf.reduce_mean(loss)
    #loss = -tf.reduce_sum(p * tf.log(q))

    #loss = tf.reduce_mean(tf.square(y_batches - x_batches + y_dist1_batches -y_dist2_batches))
    return loss

def shape_loss_batch(x_batches,y_batches, debug=False):
    loss = shape_loss_tensor(x_batches[0], y_batches[0], debug=False)
    for b in range(1,BATCH_SIZE):
        loss = tf.add(shape_loss_tensor(tf.squeeze(x_batches[b]), tf.squeeze(y_batches[b]), debug=False),loss)
    print(loss)
    loss = tf.divide(loss, float(BATCH_SIZE))
    return loss

def binary_cross_entropy(gt_tensor, pred_tensor, logits):
    # Note: tf.nn.sigmoid_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.

    # transform back to logits
    output = pred_tensor
    target = gt_tensor

    #_epsilon = tf.convert_to_tensor(EPSILON, output.dtype.base_dtype)
    #logits = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
    #output = tf.log(output / (1 - output))
    #output = tf.square(gt_tensor - pred_tensor)
    output = tf.losses.log_loss(gt_tensor,pred_tensor)#tf.nn.sigmoid_cross_entropy_with_logits(labels=gt_tensor, logits=pred_tensor) # TODO das macht probleme
    print("+++++++++++++++++++++++++")
    print(output)
    print("+++++++++++++++++++++++++")
    #output = tf.nn.sigmoid_cross_entropy_with_logits(labels=logits, logits=gt_tensor)

    return output#tf.nn.sigmoid_cross_entropy_with_logits(labels=targe, logits=output)

def binary_cross_entropy_with_distance(gt_tensor, dist1_tensor, dist2_tensor, pred_tensor, logits):
    #bce = binary_cross_entropy(gt_tensor, pred_tensor, logits)
    #bce+ w_0 * exp(-((dist1 + dist2)^2)/(2*sigma^2))
    sigma = tf.constant(5.0)
    w_0 = tf.constant(10.0)
    all_dist = tf.math.pow(tf.math.add(dist1_tensor,dist2_tensor),2)
    print(all_dist)
    exponent = tf.math.exp(tf.math.multiply(tf.math.divide(all_dist,tf.math.multiply(tf.math.pow(sigma,2),2)),-1.0))
    print(exponent)
    bce_dist = exponent#tf.math.add(bce,tf.math.multiply(w_0,exponent))
    #bce_dist = tf.reduce_mean(bce)
    return bce_dist

def test_distance_loss():
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print("Start distance loss test ...")
    distance1 = get_distance(X_SHAPE_RIGHT, 1)
    distance2 = get_distance(X_SHAPE_RIGHT, 2)
    print(distance1)

    print("#####################################")

    dist1_tensor = tf.convert_to_tensor(distance1, dtype=tf.float32)
    dist2_tensor = tf.convert_to_tensor(distance2, dtype=tf.float32)

    bce_dist_loss = binary_cross_entropy_with_distance(Y_TENSOR, dist1_tensor, dist2_tensor, X_SHAPE_RIGHT_TENSOR)

    final_bce_distance = sess.run((bce_dist_loss))
    #print("final_bce Result: "+str(final_bce))
    print("final_bce_distance Result: "+str(final_bce_distance))



def main():
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print("This is a test programm for losses...")
    loss_sw = shape_loss(X_SHAPE_WRONG, Y, SIZE,SIZE)
    loss_sr = shape_loss(X_SHAPE_RIGHT, Y, SIZE,SIZE)
    print("######################################")
    print("######################################")
    loss_sw_tensor = shape_loss_tensor(X_SHAPE_WRONG_TENSOR, Y_TENSOR, debug=False)
    loss_sr_tensor = shape_loss_tensor(X_SHAPE_RIGHT_TENSOR, Y_TENSOR, debug=False)
    print("######################################")
    print("##################NUMPY###############")
    print("Wrong: "+str(loss_sw))
    print("Right: "+str(loss_sr))
    print("######################################")
    print("#################TENSOR###############")
    print("Wrong: "+str(loss_sw_tensor))
    print("Right: "+str(loss_sr_tensor))
    print("######################################")
    print("######################################")
    '''
    print("###########SHIFT TEST#################")
    print(sess.run(SHIFT_TEST))
    shift1 = own_tensor_shift(SHIFT_TEST, SIZE, 1)
    print(sess.run(shift1))
    shift2 = own_tensor_shift(SHIFT_TEST, SIZE, 2)
    print(sess.run(shift2))
    shift3 = own_tensor_shift(SHIFT_TEST, SIZE, 3)
    print(sess.run(shift3))
    shift4 = own_tensor_shift(SHIFT_TEST, SIZE, 4)
    print(sess.run(shift4))
    '''



if __name__ == '__main__':
    #os.environ["CUDA_VISIBLE_DEVICES"]="1"
    #main()
    test_distance_loss()
