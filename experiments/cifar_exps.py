"""
This script allows you to run a host of tests on the invariant layer and
slightly different variants of it on MNIST.
"""
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision import datasets
from scatnet_learn.layers import InvariantLayerj1
import torch.nn.functional as func
import numpy as np
import random
from collections import OrderedDict
from scatnet_learn.data import cifar, tiny_imagenet
from scatnet_learn import optim
from math import sqrt

from ray.tune import Trainable

# Training settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR Example')
parser.add_argument('outdir', type=str, help='experiment directory')
parser.add_argument('--seed', type=int, default=None, metavar='S',
                    help='random seed (default: None)')
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--smoke-test', action="store_true",
                    help="Finish quickly for testing")
parser.add_argument('--datadir', type=str, default='/scratch/share/cifar',
                    help='Default location for the dataset')
parser.add_argument('--dataset', default='cifar100', type=str,
                    help='which dataset to use',
                    choices=['cifar10', 'cifar100', 'tiny_imagenet'])
parser.add_argument('--trainsize', default=-1, type=int,
                    help='size of training set')
parser.add_argument('--no-comment', action='store_true',
                    help='Turns off prompt to enter comments about run.')
parser.add_argument('--exist-ok', action='store_true',
                    help='If true, is ok if output directory already exists')
parser.add_argument('--epochs', default=120, type=int, help='num epochs')
parser.add_argument('--type', default=None, type=str, nargs='+',
                    help='Model type(s) to build')

# Core hyperparameters
parser.add_argument('--reg', default='l2', type=str, help='regularization term')
parser.add_argument('--steps', default=[60,80,100], type=int, nargs='+')
parser.add_argument('--gamma', default=0.2, type=float, help='Lr decay')


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
    """ This is a wrapper for our networks that has some useful functions"""
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
            def blk(C, F, stride, p=0.):
                if p > 0:
                    return nn.Sequential(
                        nn.Conv2d(C, F, 3, padding=1, stride=stride),
                        nn.BatchNorm2d(F),
                        nn.Dropout(p=p),
                        nn.ReLU())
                else:
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
        elif block == 'conv1x1':
            def blk(C, F, stride):
                return nn.Sequential(
                    nn.Conv2d(C, F, 1, padding=0, stride=stride),
                    nn.BatchNorm2d(F),
                    nn.ReLU())
        elif block == 'invariantj1':
            blk = InvariantLayerj1
        elif block == 'invariantj1_3x3':
            blk = lambda C, F, s: InvariantLayerj1(C, F, s, k=3)
        elif block == 'invariantj1_impulse':
            blk = lambda C, F, s: InvariantLayerj1(C, F, s, k=1, alpha='impulse')
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

    def init(self, std=1):
        """ Define the default initialization scheme """
        self.apply(net_init)
        # Try do any custom layer initializations
        for child in self.net.children():
            try:
                child.init(std)
            except AttributeError:
                pass

    def forward(self, x):
        """ Define the default forward pass"""
        out = self.net(x)
        out = self.avg(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)

        return func.log_softmax(out, dim=-1)

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

        return func.log_softmax(out, dim=-1)

# Define the options of networks. The 4 parameters are:
# (layer type, input channels, output channels, stride)
C = 64
style = {
    'A': [('conv', 3, C, 1), ('conv', C, C, 1), ('conv', C, 2*C, 2),
          ('conv', 2*C, 2*C, 1), ('conv', 2*C, 4*C, 2), ('conv', 4*C, 4*C, 1)],
    'B': [('inv', 3, C, 1), ('conv', C, C, 1), ('conv', C, 2*C, 2),
          ('conv', 2*C, 2*C, 1), ('conv', 2*C, 4*C, 2), ('conv', 4*C, 4*C, 1)],
    'C': [('conv', 3, C, 1), ('inv', C, C, 2), ('conv', C, 2*C, 1),
          ('conv', 2*C, 2*C, 1), ('conv', 2*C, 4*C, 2), ('conv', 4*C, 4*C, 1)],
    'D': [('conv', 3, C, 1), ('conv', C, C, 1), ('inv', C, 2*C, 2),
          ('conv', 2*C, 2*C, 1), ('conv', 2*C, 4*C, 2), ('conv', 4*C, 4*C, 1)],
    'E': [('conv', 3, C, 1), ('conv', C, C, 1), ('conv', C, 2*C, 2),
          ('inv', 2*C, 2*C, 2), ('conv', 2*C, 4*C, 1), ('conv', 4*C, 4*C, 1)],
    'F': [('conv', 3, C, 1), ('conv', C, C, 1), ('conv', C, 2*C, 2),
          ('conv', 2*C, 2*C, 1), ('inv', 2*C, 4*C, 2), ('conv', 4*C, 4*C, 1)],
    'G': [('conv', 3, C, 1), ('conv', C, C, 1), ('conv', C, 2*C, 2),
          ('conv', 2*C, 2*C, 1), ('conv', 2*C, 4*C, 2), ('inv', 4*C, 4*C, 1)],
    'H': [('inv', 3, C, 1), ('inv', C, C, 2), ('conv', C, 2*C, 1),
          ('conv', 2*C, 2*C, 1), ('conv', 2*C, 4*C, 2), ('conv', 4*C, 4*C, 1)],
    'I': [('conv', 3, C, 1), ('inv', C, C, 2), ('inv', C, 2*C, 1),
          ('conv', 2*C, 2*C, 1), ('conv', 2*C, 4*C, 2), ('conv', 4*C, 4*C, 1)],
    'J': [('conv', 3, C, 1), ('conv', C, C, 1), ('inv', C, 2*C, 2),
          ('inv', 2*C, 2*C, 2), ('conv', 2*C, 4*C, 1), ('conv', 4*C, 4*C, 1)],
    'K': [('conv', 3, C, 1), ('conv', C, C, 1), ('conv', C, 2*C, 2),
          ('inv', 2*C, 2*C, 2), ('inv', 2*C, 4*C, 1), ('conv', 4*C, 4*C, 1)],
    'L': [('inv', 3, C, 1), ('conv', C, C, 1), ('inv', C, 2*C, 2),
          ('conv', 2*C, 2*C, 1), ('conv', 2*C, 4*C, 2), ('conv', 4*C, 4*C, 1)],
    'M': [('conv', 3, C, 1), ('inv', C, C, 2), ('conv', C, 2*C, 1),
          ('inv', 2*C, 2*C, 2), ('conv', 2*C, 4*C, 1), ('conv', 4*C, 4*C, 1)],
    'N': [('conv', 3, C, 1), ('conv', C, C, 1), ('inv', C, 2*C, 2),
          ('conv', 2*C, 2*C, 1), ('inv', 2*C, 4*C, 2), ('conv', 4*C, 4*C, 1)],
}


class MixedNet(MyModule):
    """ MixedNet allows custom definition of conv/inv layers as you would
    a normal network. You can change the ordering below to suit your
    task
    """
    def __init__(self, dataset, type, reg='l2', wd=1e-4, wd_fc=None):
        super().__init__(dataset)
        conv = self.get_block('conv3x3')
        inv = self.get_block('invariantj1')
        layers = style[type]
        blks = []
        names = ['conv1_1', 'conv1_2', 'conv2_1',
                 'conv2_2', 'conv3_1', 'conv3_2']
        for name, (blk, C1, C2, stride) in zip(names, layers):
            if blk == 'conv':
                blks.append((name, conv(C1, C2, stride)))
            elif blk == 'inv':
                blks.append((name, inv(C1, C2, stride)))

        # F is the last output size from first 6 layers
        if dataset == 'cifar10' or dataset == 'cifar100':
            # Network is 3 stages of convolution
            self.net = nn.Sequential(OrderedDict(blks))
            self.avg = nn.AvgPool2d(8)
            self.fc1 = nn.Linear(C2, self.num_classes)
        elif dataset == 'tiny_imagenet':
            blks = blks + [
                ('conv4_1', conv(C2, 2*C2, 2)),
                ('conv4_2', conv(2*C2, 2*C2, 1))]
            self.net = nn.Sequential(OrderedDict(blks))
            self.avg = nn.AvgPool2d(8)
            self.fc1 = nn.Linear(512, self.num_classes)

        self.reg = reg
        self.wd = wd
        if wd_fc is None:
            self.wd_fc = wd
        else:
            self.wd_fc = wd_fc


class TrainNET(Trainable):
    """ This class handles model training and scheduling for our mnist networks.

    The config dictionary setup in the main function defines how to build the
    network. Then the experiment handler calles _train and _test to evaluate
    networks one epoch at a time.

    If you want to call this without using the experiment, simply ensure
    config is a dictionary with keys::

        - args: The parser arguments
        - type: The network type, a letter value between 'A' and 'N'. See above
            for what this represents.
        - lr (optional): the learning rate
        - momentum (optional): the momentum
        - wd (optional): the weight decay
        - std (optional): the initialization variance
    """
    def _setup(self, config):
        args = config.pop("args")
        vars(args).update(config)
        type_ = config.get('type')
        args.cuda = torch.cuda.is_available()

        if args.seed is not None:
            np.random.seed(args.seed)
            random.seed(args.seed)
            torch.manual_seed(args.seed)
            if args.cuda:
                torch.cuda.manual_seed(args.seed)

        # ######################################################################
        #  Data
        kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
        if args.dataset.startswith('cifar'):
            self.train_loader, self.test_loader = cifar.get_data(
                32, args.datadir, dataset=args.dataset,
                batch_size=args.batch_size, trainsize=args.trainsize,
                seed=args.seed, **kwargs)
        elif args.dataset == 'tiny_imagenet':
            self.train_loader, self.test_loader = tiny_imagenet.get_data(
                64, args.data_dir, val_only=args.testOnly,
                batch_size=args.batch_size, trainsize=args.trainsize,
                seed=args.seed, distributed=False, **kwargs)

        # ######################################################################
        # Build the network based on the type parameter. θ are the optimal
        # hyperparameters found by cross validation.
        if type_ == 'A':
            θ = (0.1, 0.9, 1e-4, 1)
        elif 'A' < type_ <= 'N':
            θ = (0.5, 0.85, 1e-4, 1)
        else:
            raise ValueError('Unknown type')

        lr, mom, wd, std = θ
        # If the parameters were provided as an option, use them
        lr = config.get('lr', lr)
        mom = config.get('mom', mom)
        wd = config.get('wd', wd)
        std = config.get('std', std)
        self.model = MixedNet(args.dataset, type_, wd=wd)
        self.model.init(std)
        self.model.cuda()

        # ######################################################################
        # Build the optimizer
        try:
            params = self.model.param_groups()
        except AttributeError:
            params = self.model.parameters()
        # Don't use the optimizer's weight decay, call that later in the loss
        # func
        self.optimizer, self.scheduler = optim.get_optim(
            'sgd', params, init_lr=lr, steps=args.steps, wd=0,
            gamma=args.gamma, momentum=mom, max_epochs=args.epochs)
        self.args = args

    def _train_iteration(self):
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = func.nll_loss(output, target)
            # Get the regularization loss directly from the network
            try:
                loss += self.model.get_reg()
            except AttributeError:
                try:
                    loss += self.model.module.get_reg()
                except AttributeError:
                    pass
            loss.backward()
            self.optimizer.step()

    def _test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                if self.args.cuda:
                    data, target = data.cuda(), target.cuda()
                output = self.model(data)
                # sum up batch loss
                test_loss += func.nll_loss(
                    output, target, reduction='sum').item()
                # get the index of the max log-probability
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(
                    target.data.view_as(pred)).long().cpu().sum()

        test_loss = test_loss / len(self.test_loader.dataset)
        accuracy = correct.item() / len(self.test_loader.dataset)
        return {"mean_loss": test_loss, "mean_accuracy": accuracy}

    def _train(self):
        self.scheduler.step()
        self._train_iteration()
        return self._test()

    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, checkpoint_path)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


if __name__ == "__main__":
    datasets.MNIST('~/data', train=True, download=True)
    args = parser.parse_args()

    import ray
    from ray import tune
    from ray.tune.schedulers import AsyncHyperBandScheduler
    from shutil import copyfile

    ray.init()
    exp_name = args.outdir
    outdir = os.path.join(os.environ['HOME'], 'ray_results', exp_name)
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    # Copy this source file to the output directory for record keeping
    copyfile(__file__, os.path.join(outdir, 'search.py'))

    sched = AsyncHyperBandScheduler(
        time_attr="training_iteration",
        reward_attr="neg_mean_loss",
        max_t=200,
        grace_period=120)

    if args.type is not None:
        type_ = args.type
    else:
        type_ = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
                 'M', 'N']

    tune.run_experiments(
        {
            exp_name: {
                "stop": {
                    #  "mean_accuracy": 0.95,
                    "training_iteration": 1 if args.smoke_test else 120,
                },
                "resources_per_trial": {
                    "cpu": 1,
                    "gpu": 0.5,
                },
                "run": TrainNET,
                #  "num_samples": 1 if args.smoke_test else 40,
                "num_samples": 5,
                "checkpoint_at_end": True,
                "config": {
                    "args": args,
                    "type": tune.grid_search(type_),
                    # "lr": tune.sample_from(lambda spec: np.random.uniform(
                    #     0.05, 1
                    # )),
                    # "mom": tune.sample_from(lambda spec: np.random.uniform(
                    #     0.75, 0.95
                    # )),
                    # "wd": tune.sample_from(lambda spec: np.random.uniform(
                    #     1e-5, 5e-4
                    # ))
                    #  "lr": tune.grid_search([0.01, 0.0316, 0.1, 0.316, 1]),
                    #  "momentum": tune.grid_search([0.7, 0.8, 0.9]),
                    #  "wd": tune.grid_search([1e-5, 1e-1e-4]),
                    #  "std": tune.grid_search([0.5, 1., 1.5, 2.0])
                }
            }
        },
        verbose=1,
        scheduler=sched)
