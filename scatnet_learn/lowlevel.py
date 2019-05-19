from __future__ import absolute_import
import torch
import torch.nn.functional as F
import numpy as np

from numpy import load
from pkg_resources import resource_stream
COEFF_CACHE = {}


def mode_to_int(mode):
    if mode == 'zero':
        return 0
    elif mode == 'symmetric':
        return 1
    elif mode == 'per' or mode == 'periodization':
        return 2
    elif mode == 'constant':
        return 3
    elif mode == 'reflect':
        return 4
    elif mode == 'replicate':
        return 5
    elif mode == 'periodic':
        return 6
    else:
        raise ValueError("Unkown pad type: {}".format(mode))


def int_to_mode(mode):
    if mode == 0:
        return 'zero'
    elif mode == 1:
        return 'symmetric'
    elif mode == 2:
        return 'periodization'
    elif mode == 3:
        return 'constant'
    elif mode == 4:
        return 'reflect'
    elif mode == 5:
        return 'replicate'
    elif mode == 6:
        return 'periodic'
    else:
        raise ValueError("Unkown pad type: {}".format(mode))


def q2c(y):
    """
    Convert from quads in y to complex numbers in z.
    """

    # Arrange pixels from the corners of the quads into
    # 2 subimages of alternate real and imag pixels.
    #  a----b
    #  |    |
    #  |    |
    #  c----d
    # Combine (a,b) and (d,c) to form two complex subimages.
    y = y/np.sqrt(2)
    a, b = y[:,:, 0::2, 0::2], y[:,:, 0::2, 1::2]
    c, d = y[:,:, 1::2, 0::2], y[:,:, 1::2, 1::2]

    return ((a-d, b+c), (a+d, b-c))


def c2q(w1, w2):
    """
    Scale by gain and convert from complex w(:,:,1:2) to real quad-numbers
    in z.

    Arrange pixels from the real and imag parts of the 2 highpasses
    into 4 separate subimages .
     A----B     Re   Im of w(:,:,1)
     |    |
     |    |
     C----D     Re   Im of w(:,:,2)

    """
    w1r, w1i = w1
    w2r, w2i = w2

    x1 = w1r + w2r
    x2 = w1i + w2i
    x3 = w1i - w2i
    x4 = -w1r + w2r

    # Get the shape of the tensor excluding the real/imagniary part
    b, ch, r, c = w1r.shape

    # Create new empty tensor and fill it
    y = w1r.new_zeros((b, ch, r*2, c*2), requires_grad=w1r.requires_grad)
    y[:, :, ::2,::2] = x1
    y[:, :, ::2, 1::2] = x2
    y[:, :, 1::2, ::2] = x3
    y[:, :, 1::2, 1::2] = x4
    y /= np.sqrt(2)

    return y


def reflect(x, minx, maxx):
    """Reflect the values in matrix *x* about the scalar values *minx* and
    *maxx*.  Hence a vector *x* containing a long linearly increasing series is
    converted into a waveform which ramps linearly up and down between *minx*
    and *maxx*.  If *x* contains integers and *minx* and *maxx* are (integers +
    0.5), the ramps will have repeated max and min samples.
    """
    x = np.asanyarray(x)
    rng = maxx - minx
    rng_by_2 = 2 * rng
    mod = np.fmod(x - minx, rng_by_2)
    normed_mod = np.where(mod < 0, mod + rng_by_2, mod)
    out = np.where(normed_mod >= rng, rng_by_2 - normed_mod, normed_mod) + minx
    return np.array(out, dtype=x.dtype)


def symm_pad_1d(l, m):
    """ Creates indices for symmetric padding. Works for 1-D.

    Inptus:
        l (int): size of input
        m (int): size of filter
    """
    xe = reflect(np.arange(-m, l+m, dtype='int32'), -0.5, l-0.5)
    return xe


def colfilter(X, h, mode='symmetric'):
    if X.shape == torch.Size([]):
        return X.new_zeros((1,1,1,1))
    ch, r = X.shape[1:3]
    m = h.shape[2] // 2
    if mode == 'symmetric':
        xe = symm_pad_1d(r, m)
        y = F.conv2d(X[:,:,xe], h.repeat(ch,1,1,1), groups=ch)
    else:
        y = F.conv2d(X, h.repeat(ch, 1, 1, 1), groups=ch, padding=(m, 0))
    return y


def rowfilter(X, h, mode='symmetric'):
    if X.shape == torch.Size([]):
        return X.new_zeros((1,1,1,1))
    ch, _, c = X.shape[1:]
    m = h.shape[2] // 2
    h = h.transpose(2,3).contiguous()
    if mode == 'symmetric':
        xe = symm_pad_1d(c, m)
        y = F.conv2d(X[:,:,:,xe], h.repeat(ch,1,1,1), groups=ch)
    else:
        y = F.conv2d(X, h.repeat(ch,1,1,1), groups=ch, padding=(0, m))
    return y


def _load_from_file(basename, varnames):

    try:
        mat = COEFF_CACHE[basename]
    except KeyError:
        with resource_stream('scatnet_learn.data', basename + '.npz') as f:
            mat = dict(load(f))
        COEFF_CACHE[basename] = mat

    try:
        return tuple(mat[k] for k in varnames)
    except KeyError:
        raise ValueError(
            'Wavelet does not define ({0}) coefficients'.format(
                ', '.join(varnames)))


def biort(name):
    """Load level 1 wavelet by name.

    :param name: a string specifying the wavelet family name
    :returns: a tuple of vectors giving filter coefficients

    =============  ============================================
    Name           Wavelet
    =============  ============================================
    antonini       Antonini 9,7 tap filters.
    farras         Farras 8,8 tap filters
    legall         LeGall 5,3 tap filters.
    near_sym_a     Near-Symmetric 5,7 tap filters.
    near_sym_b     Near-Symmetric 13,19 tap filters.
    near_sym_b_bp  Near-Symmetric 13,19 tap filters + BP filter
    =============  ============================================

    Return a tuple whose elements are a vector specifying the h0o, g0o, h1o and
    g1o coefficients.

    See :ref:`rot-symm-wavelets` for an explanation of the ``near_sym_b_bp``
    wavelet filters.

    :raises IOError: if name does not correspond to a set of wavelets known to
        the library.
    :raises ValueError: if name doesn't specify
    """
    if name == 'near_sym_b_bp':
        return _load_from_file(name, ('h0o', 'g0o', 'h1o', 'g1o', 'h2o', 'g2o'))
    else:
        return _load_from_file(name, ('h0o', 'g0o', 'h1o', 'g1o'))


def _as_col_vector(v):
    """Return *v* as a column vector with shape (N,1).
    """
    v = np.atleast_2d(v)
    if v.shape[0] == 1:
        return v.T
    else:
        return v


def prep_filt(h, c, transpose=False):
    """ Prepares an array to be of the correct format for pytorch.
    Can also specify whether to make it a row filter (set tranpose=True)"""
    h = _as_col_vector(h)[::-1]
    h = np.reshape(h, [1, 1, *h.shape])
    h = np.repeat(h, repeats=c, axis=0)
    if transpose:
        h = h.transpose((0,1,3,2))
    h = np.copy(h)
    return torch.tensor(h, dtype=torch.get_default_dtype())


class SmoothMagFn(torch.autograd.Function):
    """ Class to do complex magnitude """
    @staticmethod
    def forward(ctx, x, y, b):
        r = torch.sqrt(x**2 + y**2 + b**2)
        if x.requires_grad:
            dx = x/r
            dy = y/r
            ctx.save_for_backward(dx, dy)

        return r - b

    @staticmethod
    def backward(ctx, dr):
        dx = None
        if ctx.needs_input_grad[0]:
            drdx, drdy = ctx.saved_tensors
            dx = drdx * dr
            dy = drdy * dr
        return dx, dy, None


class ScatLayerj1_f(torch.autograd.Function):
    """ Function to do forward and backward passes of a single scattering
    layer with the DTCWT biorthogonal filters. """

    @staticmethod
    def forward(ctx, x, h0o, h1o, mode, bias):
        #  bias = 1e-2
        #  bias = 0
        ctx.in_shape = x.shape
        batch, ch, r, c = x.shape
        ctx.extra_rows = 0
        ctx.extra_cols = 0
        mode = int_to_mode(mode)
        ctx.mode = mode

        with torch.no_grad():
            # Do the single scale DTCWT
            # If the row/col count of X is not divisible by 2 then we need to
            # extend X
            if r % 2 != 0:
                x = torch.cat((x, x[:,:,-1:]), dim=2)
                ctx.extra_rows = 1
            if c % 2 != 0:
                x = torch.cat((x, x[:,:,:,-1:]), dim=3)
                ctx.extra_cols = 1

            # Level 1 forward (biorthogonal analysis filters)
            Lo = rowfilter(x, h0o, mode)
            LoHi = colfilter(Lo, h1o, mode)
            LoLo = colfilter(Lo, h0o, mode)
            LoLo = F.avg_pool2d(LoLo, 2)
            Hi = rowfilter(x, h1o, mode)
            HiLo = colfilter(Hi, h0o, mode)
            HiHi = colfilter(Hi, h1o, mode)

            # Clear up variables as we go.
            # We must be quite aggressive with this as the
            # DTCWT based scatternet allocates several times the amount of
            # needed memory. PyTorch will automatically clean it all up once we
            # return from this function, but this may be too late for large
            # tensors.
            del Lo, Hi

            # Convert quads to real and imaginary
            (deg15r, deg15i), (deg165r, deg165i) = q2c(LoHi)
            (deg45r, deg45i), (deg135r, deg135i) = q2c(HiHi)
            (deg75r, deg75i), (deg105r, deg105i) = q2c(HiLo)
            del LoHi, HiHi, HiLo

            # Convert real and imaginary to magnitude
            reals = torch.stack(
                [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=1)
            del deg15r, deg45r, deg75r, deg105r, deg135r, deg165r
            imags = torch.stack(
                [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=1)
            del deg15i, deg45i, deg75i, deg105i, deg135i, deg165i

            r = torch.sqrt(reals**2 + imags**2 + bias**2)
            if x.requires_grad:
                drdx = reals/r
                drdy = imags/r
                ctx.save_for_backward(h0o, h1o, drdx, drdy)
            else:
                ctx.save_for_backward(h0o, h1o, torch.tensor(0.),
                                      torch.tensor(0.))

            r = r - bias
            del reals, imags
            Z = torch.cat((LoLo[:, None], r), dim=1)

        return Z

    @staticmethod
    def backward(ctx, dZ):
        dX = None
        mode = ctx.mode

        if ctx.needs_input_grad[0]:
            #  h0o, h1o, θ = ctx.saved_tensors
            h0o, h1o, drdx, drdy = ctx.saved_tensors
            # Use the special properties of the filters to get the time reverse
            h0o_t = h0o
            h1o_t = h1o

            # Level 1 backward (time reversed biorthogonal analysis filters)
            dYl, dr = dZ[:,0], dZ[:,1:]
            ll = 1/4 * F.interpolate(dYl, scale_factor=2, mode="nearest")
            reals = dr * drdx
            imags = dr * drdy
            #  reals = dYm * torch.cos(θ)
            #  imags = dYm * torch.sin(θ)
            del dr
            lh = c2q((reals[:, 0], imags[:, 0]), (reals[:, 5], imags[:, 5]))
            hl = c2q((reals[:, 2], imags[:, 2]), (reals[:, 3], imags[:, 3]))
            hh = c2q((reals[:, 1], imags[:, 1]), (reals[:, 4], imags[:, 4]))
            del reals, imags

            Hi = colfilter(hh, h1o_t, mode) + colfilter(hl, h0o_t, mode)
            Lo = colfilter(lh, h1o_t, mode) + colfilter(ll, h0o_t, mode)
            del ll, lh, hl, hh
            dX = rowfilter(Hi, h1o_t, mode) + rowfilter(Lo, h0o_t, mode)
            del Lo, Hi

            if ctx.extra_rows:
                dX = dX[..., :-1, :]
            if ctx.extra_cols:
                dX = dX[..., :-1]

        return (dX,) + (None,) * 4


class ScatLayerj1_rot_f(torch.autograd.Function):
    """ Function to do forward and backward passes of a single scattering
    layer with the DTCWT biorthogonal filters. Uses the rotationally symmetric
    filters, i.e. a slightly more expensive operation."""

    @staticmethod
    def forward(ctx, x, h0o, h1o, h2o, mode, bias):
        mode = int_to_mode(mode)
        ctx.mode = mode
        #  bias = 0
        ctx.in_shape = x.shape
        batch, ch, r, c = x.shape
        ctx.extra_rows = 0
        ctx.extra_cols = 0

        # Do the single scale DTCWT
        # If the row/col count of X is not divisible by 2 then we need to
        # extend X
        if r % 2 != 0:
            x = torch.cat((x, x[:,:,-1:]), dim=2)
            ctx.extra_rows = 1
        if c % 2 != 0:
            x = torch.cat((x, x[:,:,:,-1:]), dim=3)
            ctx.extra_cols = 1

        # Level 1 forward (biorthogonal analysis filters)
        Lo = rowfilter(x, h0o, mode)
        Hi = rowfilter(x, h1o, mode)
        Ba = rowfilter(x, h2o, mode)

        LoHi = colfilter(Lo, h1o, mode)
        HiLo = colfilter(Hi, h0o, mode)
        HiHi = colfilter(Ba, h2o, mode)
        LoLo = colfilter(Lo, h0o, mode)
        LoLo = F.avg_pool2d(LoLo, 2)
        del Lo, Hi, Ba

        # Convert quads to real and imaginary
        (deg15r, deg15i), (deg165r, deg165i) = q2c(LoHi)
        (deg45r, deg45i), (deg135r, deg135i) = q2c(HiHi)
        (deg75r, deg75i), (deg105r, deg105i) = q2c(HiLo)
        del LoHi, HiHi, HiLo

        # Convert real and imaginary to magnitude
        reals = torch.stack(
            [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=1)
        del deg15r, deg45r, deg75r, deg105r, deg135r, deg165r
        imags = torch.stack(
            [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=1)
        del deg15i, deg45i, deg75i, deg105i, deg135i, deg165i

        r = torch.sqrt(reals**2 + imags**2 + bias**2)
        if x.requires_grad:
            drdx = reals/r
            drdy = imags/r
            ctx.save_for_backward(h0o, h1o, h2o, drdx, drdy)
        else:
            ctx.save_for_backward(h0o, h1o, h2o, torch.tensor(0.),
                                  torch.tensor(0.))
        r = r - bias
        del reals, imags
        Z = torch.cat((LoLo[:, None], r), dim=1)

        return Z

    @staticmethod
    def backward(ctx, dZ):
        dX = None
        mode = ctx.mode

        if ctx.needs_input_grad[0]:
            # Don't need to do time reverse as these filters are symmetric
            #  h0o, h1o, h2o, θ = ctx.saved_tensors
            h0o, h1o, h2o, drdx, drdy = ctx.saved_tensors

            # Level 1 backward (time reversed biorthogonal analysis filters)
            dYl, dr = dZ[:,0], dZ[:,1:]
            ll = 1/4 * F.interpolate(dYl, scale_factor=2, mode="nearest")

            reals = dr * drdx
            imags = dr * drdy
            #  reals = dr * torch.cos(θ)
            #  imags = dr * torch.sin(θ)
            del dr
            lh = c2q((reals[:, 0], imags[:, 0]), (reals[:, 5], imags[:, 5]))
            hl = c2q((reals[:, 2], imags[:, 2]), (reals[:, 3], imags[:, 3]))
            hh = c2q((reals[:, 1], imags[:, 1]), (reals[:, 4], imags[:, 4]))
            del reals, imags

            Lo = colfilter(lh, h1o, mode) + colfilter(ll, h0o, mode)
            Hi = colfilter(hl, h0o, mode)
            Ba = colfilter(hh, h2o, mode)
            del ll, lh, hl, hh
            dX = rowfilter(Hi, h1o, mode) + rowfilter(Lo, h0o, mode) + \
                rowfilter(Ba, h2o, mode)
            del Lo, Hi, Ba

            if ctx.extra_rows:
                dX = dX[..., :-1, :]
            if ctx.extra_cols:
                dX = dX[..., :-1]

        return (dX,) + (None,) * 5
