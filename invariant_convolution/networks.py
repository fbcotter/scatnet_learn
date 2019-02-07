import torch.nn as nn

from collections import OrderedDict
import torch.nn.init as init
from math import sqrt
import invariant_convolution.layers as l
from functools import reduce
import torch


def net_init(m):
    classname = m.__class__.__name__
    if (classname.find('Conv') != -1) or (classname.find('Linear') != -1):
        init.xavier_uniform_(m.weight, gain=sqrt(2))
        try:
            init.constant_(m.bias, 0)
        # Can get an attribute error if no bias to learn
        except AttributeError:
            pass


class MyModule(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        # Define the number of scales and classes dependent on the dataset
        if dataset == 'cifar10':
            self.num_classes = 10
            self.S = 3
        elif dataset == 'cifar100':
            self.num_classes = 100
            self.S = 3
        elif dataset == 'tiny_imagenet':
            self.num_classes = 200
            self.S = 4

    def get_block(self, block):
        """ Choose the core block type """
        if block == 'conv3x3':
            def blk(C, F, stride):
                return nn.Sequential(
                    nn.Conv2d(C, F, 3, padding=1, stride=stride),
                    nn.BatchNorm2d(F),
                    nn.ReLU())
        elif block == 'conv5x5':
            def blk(C, F, stride):
                return nn.Sequential(
                    nn.Conv2d(C, F, 5, padding=2, stride=stride),
                    nn.BatchNorm2d(F),
                    nn.ReLU())
        elif block == 'invariantj1':
            blk = l.InvariantLayerj1
        elif block == 'invariantj1c':
            blk = l.InvariantCompressLayerj1
        elif block == 'invariantj2':
            pass
        elif block == 'invariantj2c':
            pass
        elif block == 'residual':
            blk = l.ResNetBlock
        else:
            raise ValueError("Unknown block type {}".format(block))
        return blk

    def get_reg(self):
        """ Define the default regularization scheme """
        reg_loss = 0
        for param in self.net.parameters():
            if param.requires_grad:
                if self.reg == 'l1':
                    reg_loss += self.wd * torch.sum(torch.abs(param))
                else:
                    reg_loss += self.wd * torch.sum(param**2)
        for param in self.fc1.parameters():
            reg_loss += self.wd_fc * torch.sum(param**2)
        return reg_loss

    def init(self):
        """ Define the default initialization scheme """
        self.apply(net_init)
        # Try do any custom layer initializations
        for child in self.net.children():
            try:
                child.init()
            except AttributeError:
                pass

    def forward(self, x):
        """ Define the default forward pass"""
        out = self.net(x)
        out = self.avg(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)

        return out

    def forward_noise(self, x, layer=0, std=0.1):
        """ A modified forward pass where you can insert noise at a chosen
        level"""
        if layer == 0:
            x = x + std*torch.randn(x.shape, device=x.device)
            out = self.net(x)
        else:
            out = self.net[:layer](x)
            out = out + std*torch.randn(out.shape, device=out.device)
            out = self.net[layer:](out)

        out = self.avg(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)

        return out


class FlexibleNet(MyModule):
    """ FlexibleNet has a conv front end and then either a conv or invariant
    core layer for the rest of the network. You can choose how many
    conv layers are per scale. For example if you choose 2 layers per scale,
    the network would look like:

        conv
        blk

        blk
        blk

        blk
        blk
    """
    def __init__(self, dataset, block, layers_per_scale,
                 channels_per_layer, reg='l2', wd=1e-4, wd_fc=None):
        super().__init__(dataset)
        self.reg = reg
        self.wd = wd
        if wd_fc is None:
            self.wd_fc = wd
        else:
            self.wd_fc = wd_fc

        C = channels_per_layer
        Cs = [C, 2*C, 4*C, 8*C]
        if isinstance(layers_per_scale, (tuple, list)):
            Ns = layers_per_scale
        else:
            Ns = [layers_per_scale,] * self.S
        if block == 'residual':
            Ns = [n//2 for n in Ns]
        conv = self.get_block(block)

        # Create a list of layers for each scale
        scale0 = [('expand', nn.Sequential(
            nn.Conv2d(3, C, 3, padding=1), nn.BatchNorm2d(C), nn.ReLU()))]
        # scale0 = [('expand', conv(3, C, -1))]
        scale0 = scale0 + [('conv1_{}'.format(n+1), conv(C, C, 1))
                           for n in range(1, Ns[0])]
        # scales = [
        #     [('proj{}'.format(s), nn.Sequential(
        #         nn.Conv2d(Cs[s-1], Cs[s], 3, padding=1, stride=2),
        #         nn.BatchNorm2d(Cs[s]), nn.ReLU())),] +
        #     [('conv{}_{}'.format(s+1, n+1),
        #       conv(Cs[s], Cs[s], 1))
        #       for n in range(Ns[s])] for s in range(1, S)]
        scales = [
            [('conv{}_{}'.format(s+1, n+1),
              conv(Cs[s-1] if n==0 else Cs[s], Cs[s], 2 if n==0 else 1))
             for n in range(Ns[s])] for s in range(1, self.S)]

        # Create the network
        self.net = nn.Sequential(OrderedDict(
            scale0 + reduce(lambda x, y: x+y, scales))
        )
        self.avg = nn.AvgPool2d(8)
        self.fc1 = nn.Linear(Cs[self.S-1], self.num_classes)


class MixedNet(MyModule):
    """ MixedNet allows custom definition of conv/inv layers as you would
    a normal network. You can change the ordering below to suit your
    task
    """
    def __init__(self, dataset, channels, reg='l2', wd=1e-4, wd_fc=None):
        super().__init__(dataset)
        conv = self.get_block('conv3x3')
        inv = self.get_block('invariantj1')
        C = channels

        if dataset == 'cifar10' or dataset == 'cifar100':
            # Network is 3 stages of convolution
            self.net = nn.Sequential(OrderedDict([
                ('conv1_1', conv(3, C, 1)),
                ('conv1_2', conv(C, C, 1)),
                ('conv2_1', conv(C, 2*C, 2)),
                ('conv2_2', conv(2*C, 2*C, 1)),
                ('conv3_1', conv(2*C, 4*C, 2)),
                ('conv3_2', conv(4*C, 4*C, 1)),
            ]))
            self.avg = nn.AvgPool2d(8)
            self.fc1 = nn.Linear(4*C, self.num_classes)
        elif dataset == 'tiny_imagenet':
            # Network is 3 stages of convolution
            self.net = nn.Sequential(OrderedDict([
                ('conv1_1', conv(3, C, 1)),
                ('conv1_2', conv(C, C, 1)),
                ('conv2_1', conv(C, 2*C, 2)),
                ('conv2_2', conv(2*C, 2*C, 1)),
                ('conv3_1', conv(2*C, 4*C, 2)),
                ('conv3_2', conv(4*C, 4*C, 1)),
                ('conv4_1', conv(4*C, 8*C, 2)),
                ('conv4_2', conv(8*C, 8*C, 1)),
            ]))
            self.avg = nn.AvgPool2d(8)
            self.fc1 = nn.Linear(8*C, self.num_classes)
        self.reg = reg
        self.wd = wd
        if wd_fc is None:
            self.wd_fc = wd
        else:
            self.wd_fc = wd_fc


class ScatNet(MyModule):
    """ ScatNet is like a MixedNet but with a scattering front end (perhaps
    with learning between the layers)
    """
    def __init__(self, dataset, channels, reg='l2', wd=1e-4, wd_fc=None):
        super().__init__(dataset)
        conv = self.get_block('conv3x3')
        inv = self.get_block('invariantj1')
        scat = l.ScatLayer
        C = channels

        if dataset == 'cifar10' or dataset == 'cifar100':
            if dataset == 'cifar10':
                num_classes = 10
            else:
                num_classes = 100
            # Network is 3 stages of convolution
            self.net = nn.Sequential(OrderedDict([
                ('proj', conv(3, 16, 1)),
                ('scat1', scat(16, 2, learn=False, resid=False)),
                ('scat2', scat(7*16, 2, learn=False, resid=False)),
                # ('pool1', nn.MaxPool2d(2)),
                ('conv2_1', conv(49*16, 2*C, 1)),
                # ('conv2_2', conv(2*C, 2*C, 1)),
                # ('pool2', nn.MaxPool2d(2)),
                ('conv3_1', conv(2*C, 4*C, 1)),
                ('conv3_2', conv(4*C, 4*C, 1)),
            ]))
            self.avg = nn.AvgPool2d(8)
            self.fc1 = nn.Linear(4*C, num_classes)
        elif dataset == 'tiny_imagenet':
            num_classes = 200
            # Network is 3 stages of convolution
            self.net = nn.Sequential(OrderedDict([
                ('scat1', scat(3, 2, learn=True, resid=True)),
                ('scat2', scat(7*3, 2, learn=True, resid=True)),
                # ('pool1', nn.MaxPool2d(2)),
                ('conv3_1', conv(49*3, 4*C, 1)),
                ('conv3_2', conv(4*C, 4*C, 1)),
                ('pool3', nn.MaxPool2d(2)),
                ('conv4_1', conv(4*C, 8*C, 1)),
                ('conv4_2', conv(8*C, 8*C, 1)),
            ]))
            self.avg = nn.AvgPool2d(8)
            self.fc1 = nn.Linear(8*C, num_classes)
        self.reg = reg
        self.wd = wd
        if wd_fc is None:
            self.wd_fc = wd
        else:
            self.wd_fc = wd_fc

style = {
    'A': [('conv', 3, 64, 1), ('conv', 64, 64, 1), ('conv', 64, 128, 2), ('conv', 128, 128, 1), ('conv', 128, 256, 2), ('conv', 256, 256, 1)],
    'B': [('inv', 3, 64, 1), ('conv', 64, 64, 1), ('conv', 64, 128, 2), ('conv', 128, 128, 1), ('conv', 128, 256, 2), ('conv', 256, 256, 1)],
    'C': [('conv', 3, 64, 1), ('inv', 64, 64, 2), ('conv', 64, 128, 1), ('conv', 128, 128, 1), ('conv', 128, 256, 2), ('conv', 256, 256, 1)],
    'D': [('conv', 3, 64, 1), ('conv', 64, 64, 1), ('inv', 64, 128, 2), ('conv', 128, 128, 1), ('conv', 128, 256, 2), ('conv', 256, 256, 1)],
    'E': [('conv', 3, 64, 1), ('conv', 64, 64, 1), ('conv', 64, 128, 2), ('inv', 128, 128, 2), ('conv', 128, 256, 1), ('conv', 256, 256, 1)],
    'F': [('conv', 3, 64, 1), ('conv', 64, 64, 1), ('conv', 64, 128, 2), ('conv', 128, 128, 1), ('inv', 128, 256, 2), ('conv', 256, 256, 1)],
    'G': [('conv', 3, 64, 1), ('conv', 64, 64, 1), ('conv', 64, 128, 2), ('conv', 128, 128, 1), ('conv', 128, 256, 2), ('inv', 256, 256, 1)],
    'H': [('inv', 3, 64, 1), ('inv', 64, 64, 2), ('conv', 64, 128, 1), ('conv', 128, 128, 1), ('conv', 128, 256, 2), ('conv', 256, 256, 1)],
    'I': [('conv', 3, 64, 1), ('inv', 64, 64, 2), ('inv', 64, 128, 1), ('conv', 128, 128, 1), ('conv', 128, 256, 2), ('conv', 256, 256, 1)],
    'J': [('conv', 3, 64, 1), ('conv', 64, 64, 1), ('inv', 64, 128, 2), ('inv', 128, 128, 2), ('conv', 128, 256, 1), ('conv', 256, 256, 1)],
    'K': [('conv', 3, 64, 1), ('conv', 64, 64, 1), ('conv', 64, 128, 2), ('inv', 128, 128, 2), ('inv', 128, 256, 1), ('conv', 256, 256, 1)],
    'L': [('inv', 3, 64, 1), ('conv', 64, 64, 1), ('inv', 64, 128, 2), ('conv', 128, 128, 1), ('conv', 128, 256, 2), ('conv', 256, 256, 1)],
    'M': [('conv', 3, 64, 1), ('inv', 64, 64, 2), ('conv', 64, 128, 1), ('inv', 128, 128, 2), ('conv', 128, 256, 1), ('conv', 256, 256, 1)],
    'N': [('conv', 3, 64, 1), ('conv', 64, 64, 1), ('inv', 64, 128, 2), ('conv', 128, 128, 1), ('inv', 128, 256, 2), ('conv', 256, 256, 1)],
}


class MixedNet2(MyModule):
    """ MixedNet allows custom definition of conv/inv layers as you would
    a normal network. You can change the ordering below to suit your
    task
    """
    def __init__(self, dataset, type, reg='l2', wd=1e-4, wd_fc=None, ):
        super().__init__(dataset)
        conv = self.get_block('conv3x3')
        inv = self.get_block('invariantj1')
        C = 64
        layers = style[type]
        blks = []
        for name, (blk, C, F, stride) in zip(['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2'], layers):
            if blk == 'conv':
                blks.append((name, conv(C, F, stride)))
            elif blk == 'inv':
                blks.append((name, inv(C, F, stride)))

        if dataset == 'cifar10' or dataset == 'cifar100':
            # Network is 3 stages of convolution
            self.net = nn.Sequential(OrderedDict(blks))
            self.avg = nn.AvgPool2d(8)
            self.fc1 = nn.Linear(256, self.num_classes)
        elif dataset == 'tiny_imagenet':
            blks = blks + [
                ('conv4_1', conv(256, 512, 2)),
                ('conv4_2', conv(512, 512, 1))]
            self.net = nn.Sequential(OrderedDict(blks))
            self.avg = nn.AvgPool2d(8)
            self.fc1 = nn.Linear(512, self.num_classes)
        self.reg = reg
        self.wd = wd
        if wd_fc is None:
            self.wd_fc = wd
        else:
            self.wd_fc = wd_fc
