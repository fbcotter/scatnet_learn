from __future__ import absolute_import
import torch
import torch.nn.functional as F
import numpy as np

from numpy import load
from pkg_resources import resource_stream
COEFF_CACHE = {}


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

    y = torch.zeros((b, ch, r*2, c*2),
                    requires_grad=w1r.requires_grad,
                    device=w1r.device)

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


def colfilter(X, h):
    if X is None or X.shape == torch.Size([0]):
        return torch.zeros(1,1,1,1, device=X.device)
    ch, r = X.shape[1:3]
    m = h.shape[2] // 2
    xe = symm_pad_1d(r, m)
    return F.conv2d(X[:,:,xe], h.repeat(ch,1,1,1), groups=ch)


def rowfilter(X, h):
    if X is None or X.shape == torch.Size([0]):
        return torch.zeros(1,1,1,1, device=X.device)
    ch, _, c = X.shape[1:]
    m = h.shape[2] // 2
    xe = symm_pad_1d(c, m)
    h = h.transpose(2,3).contiguous()
    return F.conv2d(X[:,:,:,xe], h.repeat(ch,1,1,1), groups=ch)


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


class ScatLayerj1_f(torch.autograd.Function):
    """ Function to do forward and backward passes of a single scattering
    layer with the DTCWT biorthogonal filters. """

    @staticmethod
    def forward(ctx, x, h0o, h1o):
        bias = 1e-5
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
        Lo = rowfilter(x, h0o)
        Hi = rowfilter(x, h1o)
        LoHi = colfilter(Lo, h1o)
        HiLo = colfilter(Hi, h0o)
        HiHi = colfilter(Hi, h1o)
        LoLo = colfilter(Lo, h0o)
        LoLo = F.avg_pool2d(LoLo, 2)

        # Convert quads to real and imaginary
        (deg15r, deg15i), (deg165r, deg165i) = q2c(LoHi)
        (deg45r, deg45i), (deg135r, deg135i) = q2c(HiHi)
        (deg75r, deg75i), (deg105r, deg105i) = q2c(HiLo)

        # Convert real and imaginary to magnitude
        reals = torch.stack(
            [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=1)
        imags = torch.stack(
            [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=1)
        val = torch.sqrt(reals**2 + imags**2 + bias**2)
        mags = val - bias
        if x.requires_grad:
            θ = torch.atan2(imags, reals)
            ctx.save_for_backward(h0o, h1o, θ)
        else:
            ctx.save_for_backward(h0o, h1o, torch.tensor(0.))
        Z = torch.cat((LoLo[:, None], mags), dim=1)

        return Z

    @staticmethod
    def backward(ctx, dZ):
        dX = None

        if ctx.needs_input_grad[0]:
            h0o, h1o, θ = ctx.saved_tensors
            # Use the special properties of the filters to get the time reverse
            h0o_t = h0o
            h1o_t = h1o

            # Level 1 backward (time reversed biorthogonal analysis filters)
            dYl, dYm = dZ[:,0], dZ[:,1:]
            ll = 1/4 * F.interpolate(dYl, scale_factor=2, mode="nearest")

            reals = dYm * torch.cos(θ)
            imags = dYm * torch.sin(θ)
            lh = c2q((reals[:, 0], imags[:, 0]), (reals[:, 5], imags[:, 5]))
            hl = c2q((reals[:, 2], imags[:, 2]), (reals[:, 3], imags[:, 3]))
            hh = c2q((reals[:, 1], imags[:, 1]), (reals[:, 4], imags[:, 4]))

            Hi = colfilter(hh, h1o_t) + colfilter(hl, h0o_t)
            Lo = colfilter(lh, h1o_t) + colfilter(ll, h0o_t)
            dX = rowfilter(Hi, h1o_t) + rowfilter(Lo, h0o_t)

            if ctx.extra_rows:
                dX = dX[..., :-1, :]
            if ctx.extra_cols:
                dX = dX[..., :-1]

        return (dX,) + (None,) * 10


class ScatLayerj1_rot_f(torch.autograd.Function):
    """ Function to do forward and backward passes of a single scattering
    layer with the DTCWT biorthogonal filters. Uses the rotationally symmetric
    filters, i.e. a slightly more expensive operation."""

    @staticmethod
    def forward(ctx, x, h0o, h1o, h2o):
        bias = 1e-5
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
        Lo = rowfilter(x, h0o)
        Hi = rowfilter(x, h1o)
        Ba = rowfilter(x, h2o)

        LoHi = colfilter(Lo, h1o)
        HiLo = colfilter(Hi, h0o)
        HiHi = colfilter(Ba, h2o)
        LoLo = colfilter(Lo, h0o)
        LoLo = F.avg_pool2d(LoLo, 2)

        # Convert quads to real and imaginary
        (deg15r, deg15i), (deg165r, deg165i) = q2c(LoHi)
        (deg45r, deg45i), (deg135r, deg135i) = q2c(HiHi)
        (deg75r, deg75i), (deg105r, deg105i) = q2c(HiLo)

        # Convert real and imaginary to magnitude
        reals = torch.stack(
            [deg15r, deg45r, deg75r, deg105r, deg135r, deg165r], dim=1)
        imags = torch.stack(
            [deg15i, deg45i, deg75i, deg105i, deg135i, deg165i], dim=1)
        val = torch.sqrt(reals**2 + imags**2 + bias**2)
        mags = val - bias

        # Save info for backwards pass
        if x.requires_grad:
            θ = torch.atan2(imags, reals)
            ctx.save_for_backward(h0o, h1o, h2o, θ)

        Z = torch.cat((LoLo[:, None], mags), dim=1)

        return Z

    @staticmethod
    def backward(ctx, dZ):
        dX = None

        if ctx.needs_input_grad[0]:
            # Don't need to do time reverse as these filters are symmetric
            h0o, h1o, h2o, θ = ctx.saved_tensors

            # Level 1 backward (time reversed biorthogonal analysis filters)
            dYl, dYm = dZ[:,0], dZ[:,1:]
            ll = 1/4 * F.interpolate(dYl, scale_factor=2, mode="nearest")

            reals = dYm * torch.cos(θ)
            imags = dYm * torch.sin(θ)
            lh = c2q((reals[:, 0], imags[:, 0]), (reals[:, 5], imags[:, 5]))
            hl = c2q((reals[:, 2], imags[:, 2]), (reals[:, 3], imags[:, 3]))
            hh = c2q((reals[:, 1], imags[:, 1]), (reals[:, 4], imags[:, 4]))

            Lo = colfilter(lh, h1o) + colfilter(ll, h0o)
            Hi = colfilter(hl, h0o)
            Ba = colfilter(hh, h2o)
            dX = rowfilter(Hi, h1o) + rowfilter(Lo, h0o) + rowfilter(Ba, h2o)

            if ctx.extra_rows:
                dX = dX[..., :-1, :]
            if ctx.extra_cols:
                dX = dX[..., :-1]

        return (dX,) + (None,) * 10
