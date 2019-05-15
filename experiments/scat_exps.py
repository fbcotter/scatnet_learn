"""
This script allows you to run a host of tests on the invariant layer and
slightly different variants of it on CIFAR.
"""
from shutil import copyfile
import argparse
import os
import torch
import torch.nn as nn
import time
from scatnet_learn.layers import InvariantLayerj1, ScatLayerj1
import torch.nn.functional as func
import numpy as np
import random
from collections import OrderedDict
from scatnet_learn.data import cifar, tiny_imagenet
from scatnet_learn import optim
from tune_trainer import BaseClass, get_hms, net_init
from tensorboardX import SummaryWriter

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
parser.add_argument('--num-gpus', type=float, default=0.5)
parser.add_argument('--no-scheduler', action='store_true')
parser.add_argument('--type', default=None, type=str, nargs='+',
                    help='''Model type(s) to build.''')

# Core hyperparameters
parser.add_argument('--reg', default='l2', type=str, help='regularization term')
parser.add_argument('--steps', default=[60,80,100], type=int, nargs='+')
parser.add_argument('--gamma', default=0.2, type=float, help='Lr decay')


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
p = 0.3
nets = {'ref': [('conv', 3, 21, 0), ('pool', 1, None, None),
                ('conv', 21, 147, 0), ('pool', 2, None, None),
                ('conv', 147, 2*C, p), ('conv', 2*C, 2*C, p),
                ('conv', 2*C, 4*C, p), ('conv', 4*C, 4*C, p)],
        'scatA':[('scat', 3, None, None), ('scat', 21, None, None),
                 ('conv', 147, 2*C, p), ('conv', 2*C, 2*C, p),
                 ('conv', 2*C, 4*C, p), ('conv', 4*C, 4*C, p)],
        'scatB':[('inv', 3, 21, 2), ('inv', 21, 147, 2),
                 ('conv', 147, 2*C, p), ('conv', 2*C, 2*C, p),
                 ('conv', 2*C, 4*C, p), ('conv', 4*C, 4*C, p)],
        'scatC':[('conv', 3, 16, 0),
                 ('scat', 16, None, None), ('scat', 16*7, None, None),
                 ('conv', 16*49, 2*C, p), ('conv', 2*C, 2*C, p),
                 ('conv', 2*C, 4*C, p), ('conv', 4*C, 4*C, p)],
        'scatD':[('conv', 3, 16, 0),
                 ('inv', 16, 16*7, 2), ('inv', 16*7, 16*49, 2),
                 ('conv', 16*49, 2*C, p), ('conv', 2*C, 2*C, p),
                 ('conv', 2*C, 4*C, p), ('conv', 4*C, 4*C, p)],
        'scatD1':[('conv', 3, 16, 0),
                  ('inv', 16, 50, 2), ('inv', 50, 147, 2),
                  ('conv', 147, 2*C, p), ('conv', 2*C, 2*C, p),
                  ('conv', 2*C, 4*C, p), ('conv', 4*C, 4*C, p)],
        'scatD2':[('conv', 3, 16, 0),
                  ('inv', 16, 112, 2), ('inv', 112, 147, 2),
                  ('conv', 147, 2*C, p), ('conv', 2*C, 2*C, p),
                  ('conv', 2*C, 4*C, p), ('conv', 4*C, 4*C, p)],}


class ScatNet(nn.Module):
    """ ScatNet is like a MixedNet but with a scattering front end (perhaps
    with learning between the layers)
    """
    def __init__(self, dataset, type_, b=1e-2):
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

        # Build the selected net type by using the dictionary nets
        layers = nets[type_]
        blks = []
        layer = 0
        for typ, C1, C2, drop_p in layers:
            letter = chr(ord('A') + layer)
            if typ == 'conv':
                name = 'conv' + letter
                # Add a triple of layers for each convolutional layer
                if drop_p > 0:
                    blk = nn.Sequential(
                        nn.Conv2d(C1, C2, 3, padding=1, stride=1, bias=False),
                        nn.BatchNorm2d(C2), nn.Dropout(p=drop_p), nn.ReLU())
                else:
                    blk = nn.Sequential(
                        nn.Conv2d(C1, C2, 3, padding=1, stride=1, bias=False),
                        nn.BatchNorm2d(C2), nn.ReLU())
                layer += 1
            elif typ == 'pool':
                name = 'pool' + str(C1)
                blk = nn.MaxPool2d(2)
            elif typ == 'scat':
                name = 'scat' + letter
                blk = nn.Sequential(
                    ScatLayerj1(), nn.BatchNorm2d(7*C1), nn.ReLU())
                layer += 1
            elif typ == 'inv':
                name = 'inv' + letter
                blk = nn.Sequential(InvariantLayerj1(C1, C2),
                                    nn.BatchNorm2d(C2),
                                    nn.ReLU())
                layer += 1
            # Add the name and block to the list
            blks.append((name, blk))

        # Build the common end point
        if dataset == 'cifar10' or dataset == 'cifar100':
            # Network is 3 stages of convolution
            self.net = nn.Sequential(OrderedDict(blks))
            self.avg = nn.AvgPool2d(8)
            self.fc1 = nn.Linear(C2, self.num_classes)
        elif dataset == 'tiny_imagenet':
            # Add 3 more layers to tiny imagenet
            blk1 = nn.MaxPool2d(2)
            blk2 = nn.Sequential(
                nn.Conv2d(C2, 2*C2, 3, padding=1, stride=1, bias=False),
                nn.BatchNorm2d(2*C2), nn.ReLU())
            blk3 = nn.Sequential(
                nn.Conv2d(2*C2, 2*C2, 3, padding=1, stride=1, bias=False),
                nn.BatchNorm2d(2*C2), nn.ReLU())
            blks = blks + [
                ('pool3', blk1),
                ('convG', blk2),
                ('convH', blk3)]
            self.net = nn.Sequential(OrderedDict(blks))
            self.avg = nn.AvgPool2d(8)
            self.fc1 = nn.Linear(2*C2, self.num_classes)

    def forward(self, x):
        """ Define the default forward pass"""
        out = self.net(x)
        out = self.avg(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return func.log_softmax(out, dim=-1)


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
                64, args.datadir, val_only=args.testOnly,
                batch_size=args.batch_size, trainsize=args.trainsize,
                seed=args.seed, distributed=False, **kwargs)

        # ######################################################################
        # Build the network based on the type parameter. θ are the optimal
        # hyperparameters found by cross validation.
        if type_ == 'ref':
            θ = (0.1, 0.9, 1e-4, 1)
        else:
            θ = (0.5, 0.9, 5e-5, 1.5)
            #  raise ValueError('Unknown type')
        lr, mom, wd, std = θ
        # If the parameters were provided as an option, use them
        lr = config.get('lr', lr)
        mom = config.get('mom', mom)
        wd = config.get('wd', wd)
        std = config.get('std', std)
        #  drop_p = config.get('drop_p', drop_p)

        # Build the network
        self.model = ScatNet(args.dataset, type_, wd)
        init = lambda x: net_init(x, std)
        self.model.apply(init)

        # Split across GPUs
        if torch.cuda.device_count() > 1 and args.num_gpus > 1:
            self.model = nn.DataParallel(self.model)
            model = self.model.module
        else:
            model = self.model
        if self.use_cuda:
            self.model.cuda()

        # ######################################################################
        # Build the optimizer - use separate parameter groups for the invariant
        # and convolutional layers
        default_params = list(model.fc1.parameters())
        inv_params = []
        for name, module in model.net.named_children():
            params = [p for p in module.parameters() if p.requires_grad]
            if name.startswith('inv'):
                inv_params += params
            else:
                default_params += params

        self.optimizer, self.scheduler = optim.get_optim(
            'sgd', default_params, init_lr=lr,
            steps=args.steps, wd=wd, gamma=0.2, momentum=mom,
            max_epochs=args.epochs)

        if len(inv_params) > 0:
            # Get special optimizer parameters
            lr1 = config.get('lr1', lr)
            gamma1 = config.get('gamma1', 0.2)
            mom1 = config.get('mom1', mom)
            wd1 = config.get('wd1', wd)

            self.optimizer1, self.scheduler1 = optim.get_optim(
                'sgd', inv_params, init_lr=lr1,
                steps=args.steps, wd=wd1, gamma=gamma1, momentum=mom1,
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
        # Create reporting objects
        args.verbose = True
        outdir = os.path.join(os.environ['HOME'], 'nonray_results', args.outdir)
        tr_writer = SummaryWriter(os.path.join(outdir, 'train'))
        val_writer = SummaryWriter(os.path.join(outdir, 'val'))
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        # Copy this source file to the output directory for record keeping
        copyfile(__file__, os.path.join(outdir, 'search.py'))

        # Choose the model to run and build it
        if args.type is None:
            type_ = 'ref'
        else:
            type_ = args.type[0]
        cfg = {'args': args, 'type': type_}
        trn = TrainNET(cfg)
        trn._final_epoch = args.epochs

        # Train for set number of epochs
        elapsed_time = 0
        best_acc = 0
        for epoch in range(trn.final_epoch):
            print("\n| Training Epoch #{}".format(epoch))
            print('| Learning rate: {}'.format(
                trn.optimizer.param_groups[0]['lr']))
            print('| Momentum : {}'.format(
                trn.optimizer.param_groups[0]['momentum']))
            start_time = time.time()
            # Update the scheduler
            trn.step_lr()

            # Train for one iteration and update
            trn_results = trn._train_iteration()
            tr_writer.add_scalar('loss', trn_results['mean_loss'], epoch)
            tr_writer.add_scalar('acc', trn_results['mean_accuracy'], epoch)
            tr_writer.add_scalar('acc5', trn_results['acc5'], epoch)

            # Validate
            val_results = trn._test()
            val_writer.add_scalar('loss', val_results['mean_loss'], epoch)
            val_writer.add_scalar('acc', val_results['mean_accuracy'], epoch)
            val_writer.add_scalar('acc5', val_results['acc5'], epoch)
            acc = val_results['mean_accuracy']
            if acc > best_acc:
                print('| Saving Best model...\t\t\tTop1 = {:.2f}%'.format(acc))
                trn._save(outdir, 'model_best.pth')
                best_acc = acc

            trn._save(outdir, name='model_last.pth')
            epoch_time = time.time() - start_time
            elapsed_time += epoch_time
            print('| Elapsed time : %d:%02d:%02d\t Epoch time: %.1fs' % (
                  get_hms(elapsed_time) + (epoch_time,)))

    # We are using a scheduler

    else:

        args.verbose = False
        import ray
        from ray import tune
        from ray.tune.schedulers import AsyncHyperBandScheduler
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
        if args.type is None:
            type_ = list(nets.keys())
        else:
            type_ = args.type

        m, b = linear_func(0.1, 0.9, 0.7, 0.75)
        tune.run_experiments(
            {
                exp_name: {
                    "stop": {
                        #  "mean_accuracy": 0.95,
                        "training_iteration": (1 if args.smoke_test
                                               else args.epochs),
                    },
                    "resources_per_trial": {
                        "cpu": 1,
                        "gpu": 0 if args.cpu else args.num_gpus
                    },
                    "run": TrainNET,
                    #  "num_samples": 1 if args.smoke_test else 40,
                    "num_samples": 10 if args.nsamples == 0 else args.nsamples,
                    "checkpoint_at_end": True,
                    "config": {
                        "args": args,
                        "type": tune.grid_search(type_),
                        #  "lr": tune.sample_from(
                        #      lambda spec: np.random.uniform(0.1, 0.7)),
                        #  "mom": tune.sample_from(
                        #      lambda spec: m * spec.config.lr + b + \
                        #      0.05*np.random.randn()),
                        #  "wd": tune.sample_from(
                        #      lambda spec: np.random.uniform(1e-5, 5e-4)),
                        #  "lr": tune.grid_search([0.0316, 0.1, 0.316, 1]),
                        #  "momentum": tune.grid_search([0.7, 0.8, 0.9]),
                        #  "wd": tune.grid_search([1e-5, 1e-4]),
                        #  "std": tune.grid_search([0.5, 1., 1.5, 2.0])
                    }
                }
            },
            verbose=1,
            scheduler=sched)
