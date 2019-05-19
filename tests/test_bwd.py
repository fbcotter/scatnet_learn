from torch.autograd import gradcheck
from scatnet_learn.layers import ScatLayerj1
from scatnet_learn.lowlevel import SmoothMagFn
import torch
import pytest


@pytest.mark.parametrize('biort', ['near_sym_a', 'near_sym_b', 'near_sym_b_bp'])
def test_grad_scat(biort):
    x = torch.randn(1, 3, 32, 32, requires_grad=True, dtype=torch.double)
    scat = ScatLayerj1(biort=biort)
    scat = scat.to(torch.double)
    gradcheck(scat, (x,))


@pytest.mark.parametrize('magbias', [0, 1e-1, 1e-2, 1e-3])
def test_grad_mag(magbias):
    x = torch.randn(1, 3, 32, 32, requires_grad=True, dtype=torch.double)
    y = torch.randn(1, 3, 32, 32, requires_grad=True, dtype=torch.double)
    gradcheck(SmoothMagFn.apply, (x, y, magbias))
