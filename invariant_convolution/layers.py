import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.autograd import Function
from pytorch_wavelets import DTCWTForward, DTCWTInverse
from math import sqrt
import torch.nn.init as init


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


class Mag2Reshape(nn.Module):
    def __init__(self, ri_dim=-1):
        super().__init__()
        self.ri_dim = ri_dim

    def forward(self, x):
        r, i = torch.unbind(x, dim=self.ri_dim)
        y = r**2 + i**2
        b, _, c, h, w = y.shape
        return y.view(b, 6*c, h, w)


class MaxReshape(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        b, _, c, h, w, _ = x.shape
        y = torch.max(x[..., 0], x[..., 1])
        return y.view(b, c*6, h, w)


class RealOnly(nn.Module):
    def __init__(self, ri_dim=-1):
        super().__init__()
        self.ri_dim = ri_dim

    def forward(self, x):
        idx = torch.tensor(0, device=x.device)
        y = torch.index_select(x, self.ri_dim, idx).squeeze(dim=self.ri_dim)
        b, _, c, h, w = x.shape
        return y.view(b, c*6, h, w)


class LogMagReshape(nn.Module):
    def __init__(self, b=0.01, ri_dim=-1):
        super().__init__()
        self.b = b
        self.mag = Mag2Reshape(ri_dim=ri_dim)

    def forward(self, x):
        r = self.mag(x)
        y = 0.5 * torch.log(r + self.b)
        return y


class LogMagReshapeLearn(nn.Module):
    def __init__(self, C, b=0.01, ri_dim=-1):
        super().__init__()
        self.mag = Mag2Reshape(ri_dim=ri_dim)
        self.b = nn.Parameter(b*torch.ones(1, C, 1, 1))

    def forward(self, x):
        r = self.mag(x)
        y = 0.5 * torch.log(r + torch.abs(self.b))
        return y


class InvariantLayerj1(nn.Module):
    """ This layer is a nonlinear convolutional layer.

    It takes a wavelet transform of an image and discards the phase. This is
    the nonlinear section. It means that multiple inputs will map to the same
    output, but is nice as it allows invariance to small shifts.

    As for noise, the wavelet coefficients are energy preserving, so there
    will be no magnification.
    """
    def __init__(self, C, F, stride=1, p=0.):
        super().__init__()
        self.xfm = DTCWTForward(J=1, o_dim=1, ri_dim=2)
        self.mag = MagReshape(b=0.0001, o_dim=1, ri_dim=2)
        self.gain = nn.Conv2d(C*7, F, 1)
        self.avg = nn.AvgPool2d(2)
        if p > 0:
            self.drop = nn.Dropout(p=p)
        else:
            self.drop = None
        # self.bn = nn.BatchNorm2d(F)
        assert abs(stride) == 1 or stride == 2, "Limited resampling at the moment"
        self.stride = stride

    def forward(self, x):
        if self.stride == -1:
            x = func.interpolate(x, scale_factor=2, mode='bilinear',
                                 align_corners=False)
        yl, (yh,) = self.xfm(x)

        # Take the magnitude
        U = self.mag(yh)
        # Concatenate (Multiply lowpass by 2 to keep the DC gain 1)
        z = torch.cat((2*self.avg(yl), U), dim=1)
        y = self.gain(z)
        if self.drop is not None:
            y = self.drop(y)
        y = func.relu(y)
        if self.stride == 1:
            y = func.interpolate(y, scale_factor=2, mode='bilinear',
                                 align_corners=False)
        return y

    def init(self):
        if self.learn:
            init.xavier_uniform_(self.gain.weight, gain=sqrt(2))


class InvariantCompressLayerj1(nn.Module):
    """ This layer is a nonlinear convolutional layer.

    It takes a wavelet transform of an image and discards the phase. This is
    the nonlinear section. It means that multiple inputs will map to the same
    output, but is nice as it allows invariance to small shifts.

    """
    def __init__(self, C, F, stride=1):
        super().__init__()
        self.xfm = DTCWTForward(J=1, o_before_c=True)
        self.mag = MagReshape(b=0.0001)

        self.bp1 = nn.Conv2d(C*6, F//2, 1, bias=False)
        self.bn_bp = nn.BatchNorm2d(F//2)
        self.bp2 = nn.Conv2d(F//2, F, 1, padding=0)

        self.lp1 = nn.Conv2d(C, F, 1, padding=0, stride=2)

        self.bn = nn.BatchNorm2d(F)
        assert stride == 1 or stride == 2, "Limited resampling at the moment"
        self.stride = stride

    def forward(self, x):
        yl, (yh,) = self.xfm(x)
        yhm = self.mag(yh)
        y = self.bp1(yhm)
        y = self.bp2(func.relu(self.bn_bp(y)))

        # y = y + self.lp1(yl[:,:,::2,::2])
        y = y + self.lp1(yl)

        y = func.relu(self.bn(y))

        if self.stride == 1:
            y = func.interpolate(y, scale_factor=2, mode='bilinear',
                                 align_corners=False)
        return y


class ScatLayer(nn.Module):
    """ Do a single order scattering of the input, potentially with learning
    afterwards.

    If the input has C channels, the output will have 7*C channels, the first C
    of which are the lowpasses, then the next C are the 15 degree wavelets,
    and so on.
    """
    def __init__(self, C, stride=2, learn=False, resid=False, p=0.):
        super().__init__()
        self.xfm = DTCWTForward(J=1, o_dim=1, ri_dim=3)
        self.mag = MagReshape(b=0.0001, o_dim=1, ri_dim=3)
        self.learn = learn
        self.resid = resid
        self.avg = nn.AvgPool2d(2)
        if learn:
            self.gain = nn.Conv2d(C*7, C*7, 1)
            if p > 0:
                self.drop = nn.Dropout(p=p)
            else:
                self.drop = None
        assert abs(stride) == 1 or stride == 2, "Limited resampling at the moment"
        self.stride = stride

    def forward(self, x):
        yl, (yh,) = self.xfm(x)
        yhm = self.mag(yh)
        # Multiply by 2 to keep the DC gain constant
        y = torch.cat((2*self.avg(yl), yhm), dim=1)
        if self.learn:
            if self.resid:
                y = y + func.relu(self.gain(y))
            else:
                y = func.relu(self.gain(y))
            if self.drop is not None:
                y = self.drop(y)

        if self.stride == 1:
            y = func.interpolate(y, scale_factor=2, mode='bilinear',
                                 align_corners=False)
        return y

    def init(self):
        if self.learn:
            if self.resid:
                init.xavier_uniform_(self.gain.weight, gain=.2)
            else:
                init.xavier_uniform_(self.gain.weight, gain=1)

