###############################################################################
# Module to create the 12-tap filters that act across the orientations of the
# dtcwt.
#
###############################################################################
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import tensorflow as tf
from pkg_resources import resource_stream


def _roll_out_rows(f):
    """
    Function to copy and shift one filter 12 times
    """
    filler = np.zeros((12 - f.shape[0]))
    row1 = np.concatenate([f, filler], axis=0)
    H = np.zeros((12, 12), dtype=np.complex64)
    for i in range(12):
        H[i] = np.roll(row1, i)
    return H


# Define some filters. These must be the complex conjugates of what we use to
# generate the wanted shape (think about matched filters and what would
# maximize the inner product)
f1 = np.array([1, -1j, -1j, 1])
f2 = np.array([0.5, -1j, -1.5, -1j, 0.5])


def _get_filter(filter_num=1):
    if filter_num == 1:
        f = f1 / np.sqrt(np.sum(abs(f1)**2))
    elif filter_num == 2:
        f = f2 / np.sqrt(np.sum(abs(f2)**2))

    return f


def filter1():
    """
    Returns a 1x1x12x12 matrix filled with a simple corner and shifts of it
    """
    # Get the normalized filter coeffs
    f = _get_filter(1)
    # Reshape a 12x12 filter to 1x1x12x12
    f = np.reshape(_roll_out_rows(f), [1, 1, 12, 12])
    return tf.constant(f, tf.complex64)


def filter1_inv():
    """
    Returns a 12x12 matrix filled with a simple corner and shifts of it
    """
    H = filter1()
    return tf.conj(tf.transpose(H, [1, 0]))


def filter2():
    """
    Returns a 1x1x12x12 matrix filled with a simple corner and shifts of it
    """
    f = _get_filter(2)
    # Reshape a 12x12 filter to 1x1x12x12
    f = np.reshape(_roll_out_rows(f), [1, 1, 12, 12])
    return tf.constant(f, tf.complex64)


def filter_corners_3x3(trainable=False):
    X = np.zeros((8,3,3,12),dtype=np.complex64)

    # 4 horizontal/vertical corners
    X[0][1,1] = np.array([1,1j,1,0,0,0, 0,0,0,0,0,0])
    X[0][1,2] = np.array([2,0,0,0,0,0, 0,0,0,0,0,2])
    X[0][2,1] = np.array([0,0,2,2,0,0, 0,0,0,0,0,0])

    X[1][0,1] = np.array([0,0,2,2,0,0, 0,0,0,0,0,0])
    X[1][1,1] = np.array([0,0,0,1,1j,1, 0,0,0,0,0,0])
    X[1][1,2] = np.array([0,0,0,0,0,2, 2,0,0,0,0,0])

    X[2][0,1] = np.array([0,0,0,0,0,0, 0,0,2,2,0,0])
    X[2][1,1] = np.array([0,0,0,0,0,0, 1,1j,1,0,0,0])
    X[2][1,0] = np.array([0,0,0,0,0,2, 2,0,0,0,0,0])

    X[3][2,1] = np.array([0,0,0,0,0,0, 0,0,2,2,0,0])
    X[3][1,1] = np.array([0,0,0,0,0,0, 0,0,0,1,1j,1])
    X[3][1,0] = np.array([2,0,0,0,0,0, 0,0,0,0,0,2])

    # 4 diagonal corners
    X[4][0,2] = np.array([0,4,0,0,0,0, 0,0,0,0,0,0])
    X[4][1,1] = np.array([0,1,1j,1j,1,0, 0,0,0,0,0,0])
    X[4][2,2] = np.array([0,0,0,0,4,0, 0,0,0,0,0,0])

    X[5][0,0] = np.array([0,0,0,0,4,0, 0,0,0,0,0,0])
    X[5][1,1] = np.array([0,0,0,0,1,1j,1j,1,0,0,0,0])
    X[5][0,2] = np.array([0,0,0,0,0,0, 0,4,0,0,0,0])

    X[6][2,0] = np.array([0,0,0,0,0,0, 0,4,0,0,0,0])
    X[6][1,1] = np.array([0,0,0,0,0,0, 0,1,1j,1j,1,0])
    X[6][0,0] = np.array([0,0,0,0,0,0, 0,0,0,0,4,0])

    X[7][2,2] = np.array([0,0,0,0,0,0, 0,0,0,0,4,0])
    X[7][1,1] = np.array([1j,1,0,0,0,0, 0,0,0,0,1,1j])
    X[7][2,0] = np.array([0,4,0,0,0,0, 0,0,0,0,0,0])

    for i in range(8):
        energy = np.maximum(np.sqrt(np.sum(abs(X[i])**2, axis=(0,1,2))), 0.1)
        X[i] = X[i]/energy

    # Rearrange the filters to make them of shape [3,3,12,8]
    X = np.transpose(X, [1, 2, 3, 0])

    # These were designed for the backward pass. Would normally need to take
    # conjugate transpose of it, but since tf.nn.conv2d doesn't take transpose,
    # we only take the conjugate (and keep it transposed)
    X = np.conj(X)
    return tf.Variable(X, name='dtcwt_3x3_corners', trainable=False)


def filters_rotated():
    with resource_stream('ScatNet.filters', 'corner1.npy') as f:
        X1 = np.load(f)
    with resource_stream('ScatNet.filters', 'corner2.npy') as f:
        X2 = np.load(f)
    with resource_stream('ScatNet.filters', 'corner3.npy') as f:
        X3 = np.load(f)
    #  with resource_stream('ScatNet.filters', 'corner4.npy') as f:
        #  X4 = np.load(f)

    X = tf.Variable(np.conj(np.concatenate((X1, X2, X3), axis=-1)),
                    dtype=tf.complex64, trainable=False,
                    validate_shape=True, expected_shape=(3,3,12,36),
                    name='dtcwt_3x3_corners',)
    return X


def filters_learnable(filter_num=1, rows=12):
    # Set the starting point
    init = tf.constant(_get_filter(filter_num), dtype=tf.complex64)

    # Build one set of learnable parameters
    g = tf.get_variable('filt', initializer=init)
    g_len = g.get_shape().as_list()[0]

    # Build one row of the transform matrix
    right_zeros = tf.zeros((rows - g_len,), tf.complex64)
    h = tf.concat([g, right_zeros], axis=0)

    h_len = rows
    l = []
    for i in range(rows):
        row = tf.concat([h[h_len - i:], h[:h_len - i]], axis=0)
        l.append(row)

    H = tf.stack(l, axis=0)
    return H
