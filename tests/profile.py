import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import py3nvml
import timeit
from scatnet_learn import ScatLayerj1

parser = argparse.ArgumentParser('Profile the dwt')
parser.add_argument('--no-grad', action='store_true',
                    help='Dont calculate the gradients')
parser.add_argument('-J', type=int, default=2,
                    help='number of scales of transform to do')
parser.add_argument('--batch', default=128, type=int,
                    help='Number of images in parallel')
parser.add_argument('-C', default=3, type=int,
                    help='Number of channels')
parser.add_argument('-s', '--size', default=0, type=int,
                    help='spatial size of input')
parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                    help='which device to test')
parser.add_argument('--ref', action='store_true',
                    help='Compare to doing a similar convolution')


def reference(size, no_grad, J=2, dev='cuda'):
    # prepare transform
    x = torch.randn(*size, requires_grad=(not no_grad), device=dev)
    memx = torch.cuda.memory_allocated()
    cachedx = torch.cuda.memory_cached()

    # Do transform
    filts = torch.randn(7**J, size[1], 9, 9, device=dev)
    y = F.conv2d(x, filts, padding=4, stride=2)
    cachedy = torch.cuda.memory_cached()
    memy = torch.cuda.memory_allocated() - memx

    # Do backwards pass
    if not no_grad:
        y.backward(torch.ones_like(y))
        cached = torch.cuda.memory_cached()
        mem = torch.cuda.memory_allocated() - memy
        mem = memy - torch.cuda.memory_allocated()

    print('input tensor size: {:.1f}MiB'.format(memx / 2**20))
    print('output tensor (+ saved activations) size: {:.1f}MiB'.format(memy / 2**20))
    if not no_grad:
        print('after bwd, freed memory size: {:.1f}MiB'.format(mem / 2**20))
    print()
    print('Cached memory before transform: {:.1f}MiB'.format(cachedx / 2**20))
    print('Cached memory after transform: {:.1f}MiB'.format(cachedy / 2**20))
    if not no_grad:
        print('Cached memory after backward: {:.1f}MiB'.format(cached / 2**20))

    return y


def fwd(size, no_grad, J=2, dev='cuda'):
    # Prepare the transform
    xfm = nn.Sequential(*[ScatLayerj1() for j in range(J)]).to(dev)
    x = torch.randn(*size, requires_grad=(not no_grad), device=dev)
    memx = torch.cuda.memory_allocated()
    cachedx = torch.cuda.memory_cached()

    # Do the transform
    y = xfm(x)
    cachedy = torch.cuda.memory_cached()
    memy = torch.cuda.memory_allocated() - memx

    # Do backwards pass
    if not no_grad:
        y.backward(torch.ones_like(y))
        mem = torch.cuda.memory_allocated() - memy
        mem = memy - torch.cuda.memory_allocated()
        cached = torch.cuda.memory_cached()

    print('input tensor size: {:.1f}MiB'.format(memx / 2**20))
    print('output tensor (+ saved activations) size: {:.1f}MiB'.format(memy / 2**20))
    if not no_grad:
        print('after bwd, freed memory size: {:.1f}MiB'.format(mem / 2**20))
    print()
    print('Cached memory before transform: {:.1f}MiB'.format(cachedx / 2**20))
    print('Cached memory after transform: {:.1f}MiB'.format(cachedy / 2**20))
    if not no_grad:
        print('Cached memory after backward: {:.1f}MiB'.format(cached / 2**20))

if __name__ == "__main__":
    args = parser.parse_args()
    py3nvml.grab_gpus(1)
    if args.size > 0:
        size = (args.batch, args.C, args.size, args.size)
    else:
        size = (args.batch, args.C, 128, 128)

    if args.ref:
        y = reference(size, args.no_grad, args.J, args.device)
    else:
        y = fwd(size, args.no_grad, args.J, args.device)
