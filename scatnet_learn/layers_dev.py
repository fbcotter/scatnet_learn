import torch
import torch.nn as nn
import torch.nn.functional as func
from pytorch_wavelets.dtcwt.coeffs import biort as _biort
from pytorch_wavelets.dtcwt.lowlevel import prep_filt
from scatnet_learn.lowlevel import mode_to_int
from scatnet_learn.lowlevel import ScatLayerj1a_f
import torch.nn.init as init
import numpy as np


def random_postconv_impulse(C, F):
    """ Creates a random filter with +/- 1 in one location for a
    3x3 convolution. The idea being that we can randomly offset filters from
    each other"""
    z = torch.zeros((F, C, 3, 3))
    x = np.random.randint(-1, 2, size=(F, C))
    y = np.random.randint(-1, 2, size=(F, C))
    for i in range(F):
        for j in range(C):
            z[i, j, y[i,j], x[i,j]] = 1
    return z


def random_postconv_smooth(C, F, σ=1):
    """ Creates a random filter by shifting a gaussian with std σ. Meant to
    be a smoother version of random_postconv_impulse."""
    x = np.arange(-2, 3)
    a = 1/np.sqrt(2*np.pi*σ**2) * np.exp(-x**2/σ**2)
    b = np.outer(a, a)
    z = np.zeros((F, C, 3, 3))
    x = np.random.randint(-1, 2, size=(F, C))
    y = np.random.randint(-1, 2, size=(F, C))
    for i in range(F):
        for j in range(C):
            z[i, j] = np.roll(b, (y[i,j], x[i,j]), axis=(0,1))[1:-1,1:-1]
        z[i] /= np.sqrt(np.sum(z[i]**2))
    return torch.tensor(z, dtype=torch.float32)


class InvariantLayerj1(nn.Module):
    """ Also can be called the learnable scatternet layer.

    Takes a single order scatternet layer, and mixes the outputs to give a new
    set of outputs. You can select the style of mixing, the default being a
    single 1x1 convolutional layer, but other options include a 3x3
    convolutional mixing and a 1x1 mixing with random offsets.

    Inputs:
        C (int): The number of input channels
        F (int): The number of output channels. None by default, in which case
            the number of output channels is 7*C.
        stride (int): The downsampling factor
        k (int): The mixing kernel size
        alpha (str): A fixed kernel to increase the spatial size of the mixing.
            Can be::

                - None (no expansion),
                - 'impulse' (randomly shifts bands left/right and up/down by 1
                    pixel),
                - 'smooth' (randomly shifts a gaussian left/right and up/down
                    by 1 pixel and uses the mixing matrix to expand this.
        biort (str): which biorthogonal filters to use.

    Returns:
        y (torch.tensor): The output

    """
    def __init__(self, C, F=None, stride=2, k=1, k_lp=None,
                 biort='near_sym_a', mode='symmetric', magbias=1e-2):
        super().__init__()
        if F is None:
            F = 7*C
        if k_lp is None:
            k_lp = k

        self.scat = ScatLayerj1(biort=biort, mode=mode, magbias=magbias)
        self.stride = stride
        # Create the learned mixing weights and possibly the expansion kernel
        self.A_lp = nn.Parameter(torch.randn(F, C, k_lp, k_lp))
        self.A_bp = nn.Parameter(torch.randn(F, 6*C, k, k))
        self.b = nn.Parameter(torch.zeros(F,1,1))
        init.xavier_uniform_(self.A_lp, gain=1.5)
        init.xavier_uniform_(self.A_bp, gain=1.5)
        #  self.A = nn.Parameter(torch.randn(F, 7*C, k, k))
        #  init.xavier_uniform_(self.A, gain=1.5)
        #  self.b = nn.Parameter(torch.zeros(F))
        self.C = C
        self.F = F
        self.k = k
        self.k_lp = k_lp
        self.lp_pad = (k_lp - 1)//2
        self.bp_pad = (k - 1)//2
        self.biort = biort

    def forward(self, x):
        ll, r = self.scat(x)
        ll = func.conv2d(ll, self.A_lp, padding=self.lp_pad)
        r = func.conv2d(r, self.A_bp, padding=self.bp_pad)
        y = ll + r + self.b
        #  z = self.scat(x)
        #  y = func.conv2d(z, self.A, self.b, padding=self.bp_pad)
        if self.stride == 1:
            y = func.interpolate(y, scale_factor=2, mode='bilinear',
                                 align_corners=False)
        return y

    def extra_repr(self):
        return '{}, {}, stride={}, k={}, k_lp={}'.format(
               self.C, self.F, self.stride, self.k, self.k_lp)


class ScatLayerj1(nn.Module):
    """ Does one order of scattering at a single scale. Can be made into a
    second order scatternet by stacking two of these layers.

    Inputs:
        biort (str): the biorthogonal filters to use. if 'near_sym_b_bp' will
            use the rotationally symmetric filters. These have 13 and 19 taps
            so are quite long. They also require 7 1D convolutions instead of 6.
        x (torch.tensor): Input of shape (N, C, H, W)
        mode (str): padding mode. Can be 'symmetric' or 'zero'
        magbias (float): the magnitude bias to use for smoothing
        combine_colour (bool): if true, will only have colour lowpass and have
            greyscale bandpass

    Returns:
        y (torch.tensor): y has the lowpass and invariant U terms stacked along
            the channel dimension, and so has shape (N, 7*C, H/2, W/2). Where
            the first C channels are the lowpass outputs, and the next 6C are
            the magnitude highpass outputs.
    """
    def __init__(self, biort='near_sym_a', mode='symmetric', magbias=1e-2):
        super().__init__()
        self.biort = biort
        # Have to convert the string to an int as the grad checks don't work
        # with string inputs
        self.mode_str = mode
        self.mode = mode_to_int(mode)
        self.magbias = magbias
        h0o, _, h1o, _ = _biort(biort)
        self.h0o = torch.nn.Parameter(prep_filt(h0o, 1), False)
        self.h1o = torch.nn.Parameter(prep_filt(h1o, 1), False)
        self.lp_pool = nn.AvgPool2d(2)
        #  self.lp_pool = nn.MaxPool2d(2)

    def forward(self, x):
        # Do the single scale DTCWT
        # If the row/col count of X is not divisible by 2 then we need to
        # extend X
        _, ch, r, c = x.shape
        if r % 2 != 0:
            x = torch.cat((x, x[:,:,-1:]), dim=2)
        if c % 2 != 0:
            x = torch.cat((x, x[:,:,:,-1:]), dim=3)

        ll, r = ScatLayerj1a_f.apply(
            x, self.h0o, self.h1o, self.mode, self.magbias)
        ll = self.lp_pool(ll)
        b, _, c, h, w = r.shape
        r = r.view(b, 6*c, h, w)
        return ll, r
        #  Z = torch.cat((ll[:, None], r), dim=1)
        #  b, _, c, h, w = Z.shape
        #  Z = Z.view(b, 7*c, h, w)
        #  return Z

    def extra_repr(self):
        return "biort='{}', mode='{}', magbias={}".format(
               self.biort, self.mode_str, self.magbias)
