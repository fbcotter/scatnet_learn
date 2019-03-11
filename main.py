"""
Generic main file for creating/loading a neural network and training it.

    pre: parse arguments
    step 1: builds a model
    step 2: loads data
    step 3: creates an optimizer and gets the parameters to optimize
    step 4: loop through train/val functions
"""
from __future__ import print_function

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import py3nvml
from tensorboardX import SummaryWriter
import time
from scatnet_learn.save_exp import save_experiment_info, save_acc

import random

import os
import sys
import argparse
import numpy as np

from scatnet_learn.networks import FlexibleNet, net_init, MixedNet, ScatNet, MixedNet2
from scatnet_learn.data import cifar, tiny_imagenet
from scatnet_learn import learn, optim

parser = argparse.ArgumentParser(description='ICIP 2019 Invariant Layer '
                                             'Experiments')
parser.add_argument('exp_dir', type=str,
                    help='Output directory for the experiment')
parser.add_argument('--data_dir', type=str, default='/scratch/share/cifar',
                    help='Default location for the dataset')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--chkpt', type=str, default='best',
                    choices=['best', 'last'],
                    help='Whether to load in the last checkpoint or the best')
parser.add_argument('--testOnly', '-t', action='store_true',
                    help='Test mode with the saved model')
parser.add_argument('--num_gpus', default=1, type=int,
                    help='number of gpus to use. if more than 1, will split '
                         'the batch across multiple gpus')
parser.add_argument('--summary_freq', default=4, type=int,
                    help='number of updates of training info per epoch')
parser.add_argument('--eval_period', default=2, type=int,
                    help='after how many train epochs to run validation')
parser.add_argument('--dataset', default='cifar100', type=str,
                    help='which dataset to use',
                    choices=['cifar10', 'cifar100', 'tiny_imagenet'])
parser.add_argument('--trainsize', default=-1, type=int,
                    help='size of training set')
parser.add_argument('--no_comment', action='store_true',
                    help='Turns off prompt to enter comments about run.')
parser.add_argument('--exist_ok', action='store_true',
                    help='If true, is ok if output directory already exists')
parser.add_argument('--double_size', action='store_true',
                    help='Doubles the input size before processing')
parser.add_argument('--optim', default='sgd', type=str,
                    help='The optimizer to use')
parser.add_argument('--seed', default=-1, type=int, help='random seed')
parser.add_argument('--epochs', default=120, type=int, help='num epochs')
parser.add_argument('--iter_size', default=1, type=int,
                    help='mini-batch iterations between update steps. useful ' +
                         'for accumulating gradients.')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--verbose', action='store_true',
                    help='Make plots during training')
parser.add_argument('--gpu_select', default=None, type=int, nargs='+',
                    help='list of gpus on which to run exps')
parser.add_argument('--type', default='A', type=str,
                    help='Model type to build')

# Core hyperparameters
parser.add_argument('--lr', default=0.1, type=float,
                    help='initial learning_rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='momentum for sgd updates')
parser.add_argument('--wd', default=1e-4, type=float, help='weight decay')
parser.add_argument('--reg', default='l2', type=str, help='regularization term')
parser.add_argument('--steps', default=[60,80,100], type=int, nargs='+')
parser.add_argument('--gamma', default=0.2, type=float, help='Lr decay')

parser.add_argument('--conv_layer', default='conv3x3', type=str,
                    help='Core convolutional layer. conv3x3, conv5x5 for a '
                         'regular convolutional block with 3x3 or 5x5 spatial '
                         'support. invariantj1 for a 1 scale invariant layer, '
                         'invariantj2 for a 2 scale invariant layer. '
                         'invariantj1C/j2C are the compressed versions of '
                         'these, and residual is a residual mapping. For '
                         'this, the layers_per_scale parameter should be even.',
                    choices=['conv3x3', 'conv5x5', 'invariantj1',
                             'invariantj1c', 'invariantj2', 'invariantj2c',
                             'residual']
                    )
parser.add_argument('--layers_per_scale', default=[1,2,2], type=int, nargs='+',
                    help='number of convolutional layers before downsampling')
parser.add_argument('--channels_per_scale', default=64, type=int,
                    help='number of channels for the first scale. note that '
                         'after downsampling the number of channels is doubled')
args = parser.parse_args()
print(args)


def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    return h, m, s


# If seed was not provided, create one and seed numpy and pytorch
if args.seed < 0:
    args.seed = np.random.randint(1 << 16)
np.random.seed(args.seed)
random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.manual_seed(args.seed)

# Hyperparameter settings
py3nvml.grab_gpus(args.num_gpus, gpu_select=args.gpu_select, gpu_fraction=0.7,
                  max_procs=0)
use_cuda = torch.cuda.is_available()
best_acc = 0
start_epoch, batch_size = 1, args.batch_size


# ##############################################################################
#  Model
print('\n[Phase 1] : Model setup')
if len(args.layers_per_scale) == 1:
    args.layers_per_scale = args.layers_per_scale[0]

if args.resume:
    # Load checkpoint
    print('| Resuming from checkpoint...')
    chkpt_dir = os.path.join(args.exp_dir, 'chkpt')
    assert os.path.isdir(chkpt_dir), 'Error: No checkpoint directory found!'
    file_name = args.conv_layer
    if args.chkpt == 'best':
        checkpoint = torch.load(os.path.join(chkpt_dir, file_name + '.t7'))
    else:
        checkpoint = torch.load(os.path.join(chkpt_dir, file_name + '_latest.t7'))
    net = checkpoint['net']
    best_acc = checkpoint['acc1']
    start_epoch = checkpoint['epoch']
elif args.testOnly:
    print('\n[Test Phase] : Model setup')
    chkpt_dir = os.path.join(args.exp_dir, 'chkpt')
    assert os.path.isdir(chkpt_dir), 'Error: No checkpoint directory found!'
    file_name = args.conv_layer
    if args.chkpt == 'best':
        checkpoint = torch.load(os.path.join(chkpt_dir, file_name + '.t7'))
    else:
        checkpoint = torch.load(os.path.join(chkpt_dir, file_name + '_latest.t7'))
    net = checkpoint['net']
else:
    print('| Building net with [' + args.conv_layer+ '] core...')
    chkpt_dir = os.path.join(args.exp_dir, 'chkpt')
    # net = FlexibleNet(args.dataset, args.conv_layer, args.layers_per_scale,
    #                   args.channels_per_scale, reg=args.reg, wd=args.wd)
    # net = MixedNet(args.dataset, args.channels_per_scale, reg=args.reg,
    #                wd=args.wd)
    net = ScatNet(args.dataset, args.channels_per_scale, reg=args.reg,
                  wd=args.wd)
    # net = MixedNet2(args.dataset, args.type, reg=args.reg, wd=args.wd)
    net.init()
    file_name = args.conv_layer
    save_experiment_info(args.exp_dir, args.seed,
                         args.no_comment, args.exist_ok, net)
    print(net)

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net,
                                device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()

# ##############################################################################
#  Data
print('\n[Phase 2] : Data Preparation')
print("| Preparing dataset...")
if args.dataset.startswith('cifar'):
    trainloader, testloader = cifar.get_data(
        32, args.data_dir, args.dataset, args.batch_size, args.trainsize,
        args.seed, double_size=args.double_size)
elif args.dataset == 'tiny_imagenet':
    trainloader, testloader = tiny_imagenet.get_data(
        64, args.data_dir, val_only=args.testOnly, batch_size=args.batch_size,
        trainsize=args.trainsize, seed=args.seed, distributed=(args.num_gpus>1))

# Test only option
if args.testOnly:
    acc1, acc5 = learn.validate(testloader, net, use_cuda)
    sys.exit(0)
else:
    num_iter = len(trainloader)

# ##############################################################################
#  Optimizer
print('\n[Phase 3] : Building Optimizer')
print('| Training Epochs = ' + str(args.epochs))
print('| Initial Learning Rate = ' + str(args.lr))
print('| Optimizer = ' + str(args.optim))
tr_writer = SummaryWriter(os.path.join(args.exp_dir, 'train'))
te_writer = SummaryWriter(os.path.join(args.exp_dir, 'test'))
elapsed_time = 0
# Get the parameters to optimize
try:
    params = net.param_groups()
except AttributeError:
    params = net.parameters()

# Don't use the optimizer's weight decay, call that later in the loss func
optimizer, scheduler = optim.get_optim(args.optim, params, init_lr=args.lr,
                                       steps=args.steps, wd=0,
                                       gamma=args.gamma, momentum=args.momentum,
                                       max_epochs=args.epochs)

# ##############################################################################
#  Train
print('\n[Phase 4] : Training')
# Get one batch of validation data for logging
# x, y = next(iter(testloader))
# if use_cuda:
#     x = x.cuda()

for epoch in range(start_epoch, start_epoch+args.epochs):
    start_time = time.time()
    scheduler.step()

    learn.train(trainloader, net, criterion, optimizer, epoch, args.epochs,
                use_cuda, tr_writer, summary_freq=args.summary_freq)

    if epoch % args.eval_period == 0:
        sys.stdout.write('\n| Validating...')
        sys.stdout.flush()
        acc1, acc5 = learn.validate(testloader, net, criterion, use_cuda, epoch,
                                    te_writer)
        if acc1 > best_acc:
            print('| Saving Best model...\t\t\tTop1 = {:.2f}%\tTop5 = '
                  '{:.2f}%'.format(acc1, acc5))
            state = {
                'net': net.module if use_cuda else net,
                'acc1': acc1,
                'acc5': acc5,
                'epoch': epoch,
            }
            if not os.path.isdir(chkpt_dir):
                os.mkdir(chkpt_dir)
            save_point = os.path.join(chkpt_dir, file_name + '.t7')
            torch.save(state, save_point)
            best_acc = acc1
        # Save the last epoch's run as well
        state = {
            'net': net.module if use_cuda else net,
            'acc1': acc1,
            'acc5': acc5,
            'epoch': epoch,
        }
        if not os.path.isdir(chkpt_dir):
            os.mkdir(chkpt_dir)
        save_point = os.path.join(chkpt_dir, file_name + '_latest.t7')
        torch.save(state, save_point)

    epoch_time = time.time() - start_time
    elapsed_time += epoch_time
    print('| Elapsed time : %d:%02d:%02d\t Epoch time: %.1fs' % (
        get_hms(elapsed_time) + (epoch_time,)))

print('\n[Phase 5] : Results')
print('* Test results : Acc@1 = %.2f%%' % best_acc)
save_acc(args.exp_dir, best_acc)
