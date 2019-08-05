"""
In this module we define our own nn.modules with custom backward methods so
that we can do DeConv by calling backward on a given network. Note that these
should not be used for training, only for visualizations.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from scatnet_learn.layers import ScatLayerj1
from scatnet_learn.lowlevel import ScatLayerj2_rot_f, mode_to_int, int_to_mode
from collections import OrderedDict
from pytorch_wavelets.dtcwt.transform_funcs import fwd_j1, inv_j1
from pytorch_wavelets.dtcwt.transform_funcs import fwd_j1_rot, inv_j1_rot
from pytorch_wavelets.dtcwt.transform_funcs import fwd_j2plus, inv_j2plus
from pytorch_wavelets.dtcwt.transform_funcs import fwd_j2plus_rot, inv_j2plus_rot
from pytorch_wavelets.dtcwt.lowlevel import prep_filt
from pytorch_wavelets.dtcwt.coeffs import biort as _biort, qshift as _qshift
#  from scatnet_learn.lowlevel import int_to_mode, fwd_j1, inv_j1


class ReLU_fn(Function):
    """ Guided backrpop """
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.clamp(min=0)

    @staticmethod
    def backward(ctx, dy):
        x, = ctx.saved_tensors
        dx = dy.clone()
        dx[x < 0] = 0
        dx.clamp(min=0)
        return dx


class ReLU(nn.Module):
    def forward(self, x):
        return ReLU_fn.apply(x)


#  class ScatLayerj1_f(torch.autograd.Function):
    #  """ Function to do forward and backward passes of a single scattering
    #  layer with the DTCWT biorthogonal filters. """

    #  @staticmethod
    #  def forward(ctx, x, h0o, h1o, mode, bias):
        #  #  bias = 1e-2
        #  #  bias = 0
        #  ctx.in_shape = x.shape
        #  batch, ch, r, c = x.shape
        #  assert r % 2 == c % 2 == 0
        #  mode = int_to_mode(mode)
        #  ctx.mode = mode

        #  ll, reals, imags = fwd_j1(x, h0o, h1o, False, 1, mode)
        #  ll = F.avg_pool2d(ll, 2)
        #  r = torch.sqrt(reals**2 + imags**2 + bias**2)
        #  if x.requires_grad:
            #  drdx = reals/r
            #  drdy = imags/r
            #  ctx.save_for_backward(h0o, h1o, drdx, drdy)
        #  else:
            #  z = x.new_zeros(1)
            #  ctx.save_for_backward(h0o, h1o, z, z)

        #  r = r - bias
        #  del reals, imags
        #  Z = torch.cat((ll[:, None], r), dim=1)

        #  return Z

    #  @staticmethod
    #  def backward(ctx, dZ):
        #  dX = None
        #  mode = ctx.mode

        #  if ctx.needs_input_grad[0]:
            #  #  h0o, h1o, θ = ctx.saved_tensors
            #  h0o, h1o, drdx, drdy = ctx.saved_tensors
            #  # Use the special properties of the filters to get the time reverse
            #  h0o_t = h0o
            #  h1o_t = h1o

            #  # Level 1 backward (time reversed biorthogonal analysis filters)
            #  dYl, dr = dZ[:,0], dZ[:,1:]
            #  ll = 1/4 * F.interpolate(dYl, scale_factor=2, mode="nearest")
            #  reals = dr * drdx
            #  imags = dr * drdy

            #  dX = inv_j1(ll, reals, imags, h0o_t, h1o_t, 1, 3, 4, mode)

        #  return (dX,) + (None,) * 4


def distill_sequential(module):
    out = []
    for n, m in module.named_children():
        name = m.__class__.__name__
        if name == 'ScatLayerj1':
            out.append((n, ScatLayerj1(
                biort=m.biort, mode=m.mode_str, magbias=m.magbias)))
        elif name == 'BatchNorm2d':
            out.append((n, nn.BatchNorm2d(
                m.num_features, eps=m.eps)))
            out[-1][1].weight.data = m.weight.data
            out[-1][1].bias.data = m.bias.data
            out[-1][1].running_mean.data = m.running_mean.data
            out[-1][1].running_var.data = m.running_var.data
        elif name == 'Conv2d':
            out.append((n, nn.Conv2d(
                m.in_channels, m.out_channels, m.kernel_size,
                m.stride, m.padding, m.dilation, m.groups,
                bias=(m.bias is not None))))
            out[-1][1].weight.data = m.weight.data
            if m.bias is not None:
                out[-1][1].bias.data = m.bias.data
        elif name == 'ReLU':
            out.append((n, ReLU()))
        elif name == 'Sequential':
            out.append((n, distill_sequential(m)))
        elif name == 'Dropout':
            pass
        else:
            raise ValueError("Unkown Module {}".format(name))
    return nn.Sequential(OrderedDict(out))
