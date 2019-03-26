from torch.autograd import gradcheck
from scatnet_learn.layers import ScatLayerj1
import torch
import pytest


@pytest.mark.parametrize('biort', ['near_sym_a', 'near_sym_b', 'near_sym_b_bp'])
def test_grad_scat(biort):
    x = torch.randn(1, 3, 32, 32, requires_grad=True)
    scat = ScatLayerj1(biort=biort)
    gradcheck(scat, (x,), eps=1e-4, atol=1e-1)
