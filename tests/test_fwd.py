from dtcwt_slim.numpy import Transform2d
from scatnet_learn.layers import ScatLayerj1
import numpy as np
import torch
import torch.nn.functional as F
import pytest


@pytest.mark.parametrize('biort', ['near_sym_a', 'near_sym_b', 'near_sym_b_bp'])
def test_equal(biort):
    b = 1e-5

    scat = ScatLayerj1(biort=biort)
    xfm = Transform2d(biort=biort)
    x = torch.randn(3, 4, 32, 32)
    z = scat(x)

    X = x.data.numpy()
    Yl, Yh = xfm.forward(X, nlevels=1)
    yl = torch.tensor(Yl)
    yl2 = F.avg_pool2d(yl, 2)

    M = np.sqrt(Yh[0].real**2 + Yh[0].imag**2 + b**2) - b
    M = M.transpose(0, 2, 1, 3, 4)
    m = torch.tensor(M)
    m2 = m.view(3, 24, 16, 16)
    z2 = torch.cat((yl2, m2), dim=1)
    np.testing.assert_array_almost_equal(z, z2, decimal=4)
