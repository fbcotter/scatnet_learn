###############################################################################
# Module to create the 12-tap filters that act across the orientations of the
# dtcwt.
#
###############################################################################
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import torch
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


def filters_rotated():
    with resource_stream('scatnet_learn.filters', 'corner1.npy') as f:
        X1 = np.load(f).transpose(3,2,0,1)
    with resource_stream('scatnet_learn.filters', 'corner2.npy') as f:
        X2 = np.load(f).transpose(3,2,0,1)
    with resource_stream('scatnet_learn.filters', 'corner3.npy') as f:
        X3 = np.load(f).transpose(3,2,0,1)
    #  with resource_stream('ScatNet.filters', 'corner4.npy') as f:
        #  X4 = np.load(f)
    X = np.concatenate((X1, X2, X3), axis=0)
    Xr = torch.from_numpy(X.real).to(torch.float)
    Xi = torch.from_numpy(-X.imag).to(torch.float)
    return Xr, Xi
