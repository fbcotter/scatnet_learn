"""
This script allows you to run a host of tests on the invariant layer and
slightly different variants of it on CIFAR.
"""
import sys
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.init as init
import time
from scatnet_learn.layers import InvariantLayerj1, InvariantLayerj1_compress
import torch.nn.functional as func
import numpy as np
import random
from collections import OrderedDict
from scatnet_learn.data import cifar, tiny_imagenet
from scatnet_learn import optim
from math import sqrt
from tune_trainer import BaseClass, get_hms

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
parser.add_argument('--nsamples', type=int, default=0,
                    help='The number of runs to test.')
parser.add_argument('--exist-ok', action='store_true',
                    help='If true, is ok if output directory already exists')
parser.add_argument('--epochs', default=120, type=int, help='num epochs')
parser.add_argument('--cpu', action='store_true', help='Do not run on gpus')
parser.add_argument('--no-scheduler', action='store_true')
parser.add_argument('--type', default=None, type=str, nargs='+',
                    help='''Model type(s) to build. If left blank, will run 14
networks consisting of those defined by the dictionary "nets" (0, 1, or 2
invariant layers at different depths). Can also specify to run "nets1" or
"nets2", which swaps out the invariant layers for other iterations.
Alternatively can directly specify the layer name, e.g. "invA", or "invB2".''')

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


# Define the options of networks. The 4 parameters are:
# (layer type, input channels, output channels, stride)
#
# The dictionary 'nets' has 14 different layouts of vgg nets networks with 0,
# 1 or 2 invariant layers at different depths.
# The dicionary 'nets2' is the same as 'nets' except we change the invariant
# layer for an invariant layer with random shifts
# The dicionary 'nets3' is the same as 'nets' except we change the invariant
# layer for an invariant layer with a 3x3 convolution
C = 96
nets = {
    'invA': [('inv', 3, C, 1), ('conv', C, C, 1), ('pool', 1, None, None),
             ('conv', C, 2*C, 1), ('conv', 2*C, 2*C, 1),('pool', 2, None, None),
             ('conv', 2*C, 4*C, 1), ('conv', 4*C, 4*C, 1)],
    'invB': [('conv', 3, C, 1), ('inv', C, C, 2),
             ('conv', C, 2*C, 1), ('conv', 2*C, 2*C, 1),('pool', 1, None, None),
             ('conv', 2*C, 4*C, 1), ('conv', 4*C, 4*C, 1)],
    'invC': [('conv', 3, C, 1), ('conv', C, C, 1),
             ('inv', C, 2*C, 2), ('conv', 2*C, 2*C, 1),('pool', 1, None, None),
             ('conv', 2*C, 4*C, 1), ('conv', 4*C, 4*C, 1)],
    'invD': [('conv', 3, C, 1), ('conv', C, C, 1),('pool', 1, None, None),
             ('conv', C, 2*C, 1), ('inv', 2*C, 2*C, 2),
             ('conv', 2*C, 4*C, 1), ('conv', 4*C, 4*C, 1)],
    'invE': [('conv', 3, C, 1), ('conv', C, C, 1),('pool', 1, None, None),
             ('conv', C, 2*C, 1), ('conv', 2*C, 2*C, 1),
             ('inv', 2*C, 4*C, 2), ('conv', 4*C, 4*C, 1)],
    'invF': [('conv', 3, C, 1), ('conv', C, C, 1),('pool', 1, None, None),
             ('conv', C, 2*C, 1), ('conv', 2*C, 2*C, 1),('pool', 2, None, None),
             ('conv', 2*C, 4*C, 1), ('inv', 4*C, 4*C, 1)],
    'invAB': [('inv', 3, C, 1), ('inv', C, C, 2),
              ('conv', C, 2*C, 1), ('conv', 2*C, 2*C, 1),('pool', 1, None, None),
              ('conv', 2*C, 4*C, 1), ('conv', 4*C, 4*C, 1)],
    'invBC': [('conv', 3, C, 1), ('inv', C, C, 2),
              ('inv', C, 2*C, 1), ('conv', 2*C, 2*C, 1),('pool', 1, None, None),
              ('conv', 2*C, 4*C, 1), ('conv', 4*C, 4*C, 1)],
    'invCD': [('conv', 3, C, 1), ('conv', C, C, 1),
              ('inv', C, 2*C, 2), ('inv', 2*C, 2*C, 2),
              ('conv', 2*C, 4*C, 1), ('conv', 4*C, 4*C, 1)],
    'invDE': [('conv', 3, C, 1), ('conv', C, C, 1),('pool', 1, None, None),
              ('conv', C, 2*C, 1), ('inv', 2*C, 2*C, 2),
              ('inv', 2*C, 4*C, 1), ('conv', 4*C, 4*C, 1)],
    'invAC': [('inv', 3, C, 1), ('conv', C, C, 1),
              ('inv', C, 2*C, 2), ('conv', 2*C, 2*C, 1),('pool', 2, None, None),
              ('conv', 2*C, 4*C, 1), ('conv', 4*C, 4*C, 1)],
    'invBD': [('conv', 3, C, 1), ('inv', C, C, 2),
              ('conv', C, 2*C, 1), ('inv', 2*C, 2*C, 2),
              ('conv', 2*C, 4*C, 1), ('conv', 4*C, 4*C, 1)],
    'invCE': [('conv', 3, C, 1), ('conv', C, C, 1),
              ('inv', C, 2*C, 2), ('conv', 2*C, 2*C, 1),
              ('inv', 2*C, 4*C, 2), ('conv', 4*C, 4*C, 1)],
}


def changelayer(layers, suffix='_3x3'):
    out = []
    for l in layers:
        if l[0] == 'inv':
            out.append((l[0]+suffix, l[1], l[2], l[3]))
        else:
            out.append(l)
    return out


nets1 = {k + '1': changelayer(v, '_imp') for k, v in nets.items()}
nets2 = {k + '2': changelayer(v, '_3x3') for k, v in nets.items()}
allnets = {
    'ref': [('conv', 3, C, 1), ('conv', C, C, 1),
            ('conv', C, 2*C, 2), ('conv', 2*C, 2*C, 1),
            ('conv', 2*C, 4*C, 2), ('conv', 4*C, 4*C, 1)],
    'ref2': [('conv', 3, C, 1), ('conv', C, C, 1), ('pool', 1, None, None),
             ('conv', C, 2*C, 1), ('conv', 2*C, 2*C, 1), ('pool', 2, None, None),
             ('conv', 2*C, 4*C, 1), ('conv', 4*C, 4*C, 1)],
    #  'invall': [('inv', 3, C, 1), ('inv', C, C, 1),
               #  ('inv', C, 2*C, 2), ('inv', 2*C, 2*C, 1),
               #  ('inv', 2*C, 4*C, 2), ('inv', 4*C, 4*C, 1)],
    **nets, **nets1, **nets2
}


class MixedNet(MyModule):
    """ MixedNet allows custom definition of conv/inv layers as you would
    a normal network. You can change the ordering below to suit your
    task
    """
    def __init__(self, dataset, type, biort='near_sym_a'):
        super().__init__(dataset)
        layers = allnets[type]
        blks = []
        layer = 0
        for blk, C1, C2, stride in layers:
            if blk == 'conv':
                name = 'conv' + chr(ord('A') + layer)
                # Add a triple of layers for each convolutional layer
                blk = nn.Sequential(
                    nn.Conv2d(C1, C2, 3, padding=1, stride=stride),
                    nn.BatchNorm2d(C2),
                    nn.ReLU())
                layer += 1
            elif blk == 'pool':
                name = 'pool' + str(C1)
                blk = nn.MaxPool2d(2)
            elif blk == 'inv':
                name = 'inv' + chr(ord('A') + layer)
                # Add a triple of layers for each invariant layer
                blk = nn.Sequential(
                    InvariantLayerj1(C1, C2, stride, biort=biort),
                    nn.BatchNorm2d(C2),
                    nn.ReLU())
                layer += 1
            elif blk == 'inv_imp':
                name = 'inv_imp' + chr(ord('A') + layer)
                # Add a triple of layers for each invariant layer
                blk = nn.Sequential(
                    InvariantLayerj1(
                        C1, C2, stride, alpha='impulse', biort=biort),
                    nn.BatchNorm2d(C2),
                    nn.ReLU())
                layer += 1
            elif blk == 'inv_3x3':
                name = 'inv3x3' + chr(ord('A') + layer)
                # Add a triple of layers for each invariant layer
                blk = nn.Sequential(
                    InvariantLayerj1(
                        C1, C2, stride, k=3, biort=biort),
                    nn.BatchNorm2d(C2),
                    nn.ReLU())
                layer += 1
            # Add the name and block to the list
            blks.append((name, blk))

        # C2 is the last output size from first 6 layers
        if dataset == 'cifar10' or dataset == 'cifar100':
            # Network is 3 stages of convolution
            self.net = nn.Sequential(OrderedDict(blks))
            self.avg = nn.AvgPool2d(8)
            self.fc1 = nn.Linear(C2, self.num_classes)
        elif dataset == 'tiny_imagenet':
            # Add 3 more layers to tiny imagenet
            blk1 = nn.MaxPool2d(2)
            blk2 = nn.Sequential(
                nn.Conv2d(C2, 2*C2, 3, padding=1, stride=1),
                nn.BatchNorm2d(2*C2),
                nn.ReLU())
            blk3 = nn.Sequential(
                nn.Conv2d(2*C2, 2*C2, 3, padding=1, stride=1),
                nn.BatchNorm2d(2*C2),
                nn.ReLU())
            blks = blks + [
                ('pool3', blk1),
                ('convG', blk2),
                ('convH', blk3)]
            self.net = nn.Sequential(OrderedDict(blks))
            self.avg = nn.AvgPool2d(8)
            self.fc1 = nn.Linear(2*C2, self.num_classes)


class TrainNET(BaseClass):
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
        if hasattr(args, 'verbose'):
            self._verbose = args.verbose

        if args.seed is not None:
            np.random.seed(args.seed)
            random.seed(args.seed)
            torch.manual_seed(args.seed)
            if self.use_cuda:
                torch.cuda.manual_seed(args.seed)

        # ######################################################################
        #  Data
        kwargs = {'num_workers': 0, 'pin_memory': True} if self.use_cuda else {}
        if args.dataset.startswith('cifar'):
            self.train_loader, self.test_loader = cifar.get_data(
                32, args.datadir, dataset=args.dataset,
                batch_size=args.batch_size, trainsize=args.trainsize,
                seed=args.seed, **kwargs)
        elif args.dataset == 'tiny_imagenet':
            self.train_loader, self.test_loader = tiny_imagenet.get_data(
                64, args.datadir, val_only=False,
                batch_size=args.batch_size, trainsize=args.trainsize,
                seed=args.seed, distributed=False, **kwargs)

        # ######################################################################
        # Build the network based on the type parameter. θ are the optimal
        # hyperparameters found by cross validation.
        if type_.startswith('ref'):
            θ = (0.1, 0.9, 1e-4, 1)
        elif type_ in nets.keys():
            θ = (0.5, 0.75, 1e-4, 1)
        elif type_ in nets1.keys():
            θ = (0.5, 0.85, 1e-4, 1)
        elif type_ in nets2.keys():
            θ = (0.2, 0.90, 1e-4, 1)
        elif type_ == 'invall':
            θ = (0.5, 0.70, 1e-4, 1)
        else:
            θ = (0.5, 0.85, 1e-4, 1)
            #  raise ValueError('Unknown type')
        lr, mom, wd, std = θ
        # If the parameters were provided as an option, use them
        lr = config.get('lr', lr)
        mom = config.get('mom', mom)
        wd = config.get('wd', wd)
        std = config.get('std', std)
        biort = config.get('biort', 'near_sym_a')

        # Build the network
        self.model = MixedNet(args.dataset, type_, biort=biort)
        self.model.init(std)
        if self.use_cuda:
            self.model.cuda()

        # ######################################################################
        # Build the optimizer - use separate parameter groups for the invariant
        # and convolutional layers
        default_params = {'params': list(self.model.fc1.parameters()),
                          'lr': lr, 'mom': mom, 'wd': wd}
        inv_params = {'params': [], 'lr': lr, 'mom': mom, 'wd': wd}
        for name, module in self.model.net.named_children():
            if name.startswith('inv'):
                inv_params['params'] += list(module.parameters())
            else:
                default_params['params'] += list(module.parameters())

        self.optimizer, self.scheduler = optim.get_optim(
            'sgd', [default_params, inv_params], init_lr=0.1,
            steps=args.steps, wd=1e-4, gamma=args.gamma, momentum=0.9,
            max_epochs=args.epochs)

        if self.verbose:
            print(self.model)


def linear_func(x1, y1, x2, y2):
    m = (y2-y1)/(x2-x1)
    b = y1 - m*x1
    return m, b


if __name__ == "__main__":
    args = parser.parse_args()

    if args.no_scheduler:
        args.verbose = True
        outdir = os.path.join(os.environ['HOME'], 'nonray_results', args.outdir)
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        if args.type is None:
            type_ = 'ref2'
        else:
            type_ = args.type[0]
        cfg = {'args': args, 'type': type_, 'biort': 'near_sym_a'}
        trn = TrainNET(cfg)
        elapsed_time = 0

        best_acc = 0
        for epoch in range(120):
            print('| Learning rate: {}'.format(trn.optimizer.param_groups[0]['lr']))
            print('| Momentum : {}'.format(trn.optimizer.param_groups[0]['momentum']))
            start_time = time.time()
            trn._train()
            results = trn._test()
            acc1 = results['mean_accuracy']
            if acc1 > best_acc:
                print('| Saving Best model...\t\t\tTop1 = {:.2f}%'.format(acc1))
                trn._save(outdir)
            best_acc = acc1

        epoch_time = time.time() - start_time
        elapsed_time += epoch_time
        print('| Elapsed time : %d:%02d:%02d\t Epoch time: %.1fs' % (
              get_hms(elapsed_time) + (epoch_time,)))

    else:

        args.verbose = False
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

        # Select which networks to run
        if args.type is not None:
            if len(args.type) == 1 and args.type[0] == 'nets':
                type_ = list(nets.keys()) + ['ref2']
            elif len(args.type) == 1 and args.type[0] == 'nets1':
                type_ = list(nets1.keys()) + ['ref2']
            elif len(args.type) == 1 and args.type[0] == 'nets2':
                type_ = list(nets2.keys()) + ['ref2']
            elif len(args.type) == 1 and args.type[0] == 'all':
                type_ = list(allnets.keys())
            else:
                type_ = args.type
        else:
            type_ = list(nets.keys()) + ['ref2',]

        m, b = linear_func(0.1, 0.9, 0.7, 0.75)
        if args.dataset.startswith('cifar'):
            gpus = 0.5
        else:
            gpus = 1
        tune.run_experiments(
            {
                exp_name: {
                    "stop": {
                        #  "mean_accuracy": 0.95,
                        "training_iteration": 1 if args.smoke_test else args.epochs,
                    },
                    "resources_per_trial": {
                        "cpu": 1,
                        "gpu": 0 if args.cpu else gpus
                    },
                    "run": TrainNET,
                    #  "num_samples": 1 if args.smoke_test else 40,
                    "num_samples": 10 if args.nsamples == 0 else args.nsamples,
                    "checkpoint_at_end": True,
                    "config": {
                        "args": args,
                        "type": tune.grid_search(type_),
                        #  "lr": tune.sample_from(lambda spec: np.random.uniform(
                            #  0.1, 0.7
                        #  )),
                        #  "mom": tune.sample_from(
                            #  lambda spec: m*spec.config.lr + b +
                                #  0.05*np.random.randn()),
                        #  "wd": tune.sample_from(lambda spec: np.random.uniform(
                           #  1e-5, 5e-4
                        #  ))
                        #  "lr": tune.grid_search([0.01, 0.0316, 0.1, 0.316, 1]),
                        #  "momentum": tune.grid_search([0.7, 0.8, 0.9]),
                        #  "wd": tune.grid_search([1e-5, 1e-1e-4]),
                        #  "std": tune.grid_search([0.5, 1., 1.5, 2.0])
                    }
                }
            },
            verbose=1,
            scheduler=sched)

