import torch
import torch.nn as nn
import torch.nn.functional as func
from pytorch_wavelets import DTCWTForward
from math import sqrt
import torch.nn.init as init
from scipy.fftpack import idct
import numpy as np


class SmoothMagFn(torch.autograd.Function):
    """ Class to do complex magnitude """
    @staticmethod
    def forward(ctx, x, b, ri_dim):
        ctx.ri_dim = ri_dim
        x1, x2 = torch.unbind(x, dim=ri_dim)
        val = torch.sqrt(x1**2 + x2**2 + b)
        mag = val - sqrt(b)
        if x.requires_grad:
            dx1 = x1/val
            dx2 = x2/val
            ctx.save_for_backward(dx1, dx2)

        return mag

    @staticmethod
    def backward(ctx, dy):
        dx = None
        if ctx.needs_input_grad[0]:
            dx1, dx2 = ctx.saved_tensors
            dx = torch.stack((dy*dx1, dy*dx2), dim=ctx.ri_dim)
        return dx, None, None


class MagReshape(nn.Module):
    """ Takes a smooth magnitude but also reshapes the DTCWT bandpass
    coefficients, stacking the modulus terms along the channel dimension.

    Inputs:
        b (float): the smoothing factor for the magnitude. smaller is less
            smooth
        o_dim (int): the dimension of the orientations for dtcwt outputs. should
            match the o_dim term used in the DTCWT class initialization.
        ri_dim (int): the dimension of the real and imaginary parts for the
            bandpass terms
        x (torch.tensor): input of shape (N, 6, C, H, W, 2) or similar,
            depending on o_dim and ri_dim.

    Returns:
        y (torch.tensor): output of shape (N, 6*C, H, W)
    """
    def __init__(self, b=0.01, o_dim=1, ri_dim=-1):
        super().__init__()
        self.b = b
        self.ri_dim = ri_dim % 6
        assert 1 <= o_dim <= 2, "Restricted support for the orientation " \
                                "dimension"
        self.o_dim = o_dim

    def forward(self, x):
        mag = SmoothMagFn.apply(x, self.b, self.ri_dim)
        if self.o_dim == 1:
            b, _, c, h, w = mag.shape
        else:
            b, c, _, h, w = mag.shape
        return mag.view(b, 6*c, h, w)


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


def dct_bases():
    """ Get the top 3 dct bases """
    x = np.zeros((1,1,3,3))
    x[0,0,0,0] = 1/9
    lp = idct(idct(x, axis=-2, norm='ortho'), axis=-1, norm='ortho')
    x[0,0,0,0] = 0
    x[0,0,0,1] = 1/9
    horiz = idct(idct(x, axis=-2, norm='ortho'), axis=-1, norm='ortho')
    x[0,0,0,1] = 0
    x[0,0,1,0] = 1/9
    vertic = idct(idct(x, axis=-2, norm='ortho'), axis=-1, norm='ortho')

    return (torch.tensor(lp, dtype=torch.float32),
            torch.tensor(horiz, dtype=torch.float32),
            torch.tensor(vertic, dtype=torch.float32))


class ScatLayerj1(nn.Module):
    """ Does one order of scattering at a single scale. Can be made into a
    second order scatternet by stacking two of these layers.

    Inputs:
        stride (int): 1 or 2. The downsampling factor. By default, 2 makes sense
          as the information content will be at a lower frequency from the
          scatternet output, but can use 1 if you do not wish to downsample.
        x (torch.tensor): Input of shape (N, C, H, W)

    Returns:
        y (torch.tensor): y has the lowpass and invariant U terms stacked along
            the channel dimension, and so has shape (N, 7*C, H/2, W/2). Where
            the first C channels are the lowpass outputs, and the next 6C are
            the magnitude highpass outputs.
    """

    def __init__(self, stride=2):
        super().__init__()
        self.xfm = DTCWTForward(J=1, o_dim=1, ri_dim=2)
        self.mag = MagReshape(b=1e-5, o_dim=1, ri_dim=2)
        if stride == 2:
            self.lp = nn.AvgPool2d(2)
            self.bp = lambda x: x
            self.avg = lambda x: 2*func.avg_pool2d(x, 2)
        elif stride == 1:
            self.lp = lambda x: x
            self.bp = lambda x: 0.5*func.interpolate(
                x, scale_factor=2, mode='bilinear', align_corners=False)
        else:
            raise ValueError("Can only do 1 or 2 stride")

    def forward(self, x):
        yl, (yh,) = self.xfm(x)
        # Take the magnitude
        U = self.mag(yh)
        # Concatenate
        z = torch.cat((self.lp(yl), self.bp(U)), dim=1)
        return z


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

    Returns:
        y (torch.tensor): The output

    """
    def __init__(self, C, F=None, stride=2, k=1, alpha=None):
        super().__init__()
        if F is None:
            F = 7*C
        if k > 1 and alpha is not None:
            raise ValueError("Only use alpha when k=1")

        self.scat = ScatLayerj1(stride=stride)
        # Create the learned mixing weights and possibly the expansion kernel
        self.A = nn.Parameter(torch.randn(F, 7*C, k, k))
        self.b = nn.Parameter(torch.zeros(F,))
        if alpha == 'impulse':
            self.alpha = nn.Parameter(
                random_postconv_impulse(7*C, F), requires_grad=False)
            self.pad = 1
        elif alpha == 'smooth':
            self.alpha = nn.Parameter(
                random_postconv_smooth(7*C, F, σ=1), requires_grad=False)
            self.pad = 1
        elif alpha == 'random':
            self.alpha = nn.Parameter(
                torch.randn(F, 7*C, 3, 3), requires_grad=False)
            init.xavier_uniform(self.alpha)
            self.pad = 1
        elif alpha is None:
            self.alpha = 1
            self.pad = (k-1) // 2
        else:
            raise ValueError

    def forward(self, x):
        z = self.scat(x)
        As = self.A * self.alpha
        y = func.conv2d(z, As, self.b, padding=self.pad)
        y = func.relu(y)
        return y

    def init(self, gain=1, method='xavier_uniform'):
        if method == 'xavier_uniform':
            init.xavier_uniform_(self.A, gain=gain)
        else:
            init.xavier_normal_(self.A, gain=gain)


class InvariantLayerj1_dct(nn.Module):
    """ Also can be called the learnable scatternet layer.

    Takes a single order scatternet layer, and mixes the outputs to give a new
    set of outputs. This version expands the spatial support of the mixing by
    taking the top 3 dct coefficients and learning 3 1x1 mixing matrices

    Inputs:
        C (int): The number of input channels
        F (int): The number of output channels. None by default, in which case
            the number of output channels is 7*C.
        stride (int): The downsampling factor

    Returns:
        y (torch.tensor): The output

    """
    def __init__(self, C, F, stride=2):
        super().__init__()
        self.scat = ScatLayerj1(stride=stride)
        self.A1 = nn.Parameter(torch.randn(C*7, F, 1, 1))
        self.A2 = nn.Parameter(torch.randn(C*7, F, 1, 1))
        self.A3 = nn.Parameter(torch.randn(C*7, F, 1, 1))
        self.b = nn.Parameter(torch.zeros(F,1,1))
        lp, h, v = dct_bases()
        self.lp = nn.Parameter(lp, requires_grad=False)
        self.h = nn.Parameter(h, requires_grad=False)
        self.v = nn.Parameter(v, requires_grad=False)

    def forward(self, x):
        A1 = self.A1 * self.lp
        A2 = self.A2 * self.h
        A3 = self.A3 * self.v
        z = self.scat(x)
        y = (func.conv2d(z, A1, padding=1) +
             func.conv2d(z, A2, padding=1) +
             func.conv2d(z, A3, padding=1) + self.b)
        y = func.relu(y)
        return y

    def init(self, gain=1, method='xavier_uniform'):
        init.xavier_uniform_(self.A1, gain=gain)
        init.xavier_uniform_(self.A2, gain=gain)
        init.xavier_uniform_(self.A3, gain=gain)
