"""
This script allows you to run a host of tests on the invariant layer and
slightly different variants of it on MNIST.
"""
import argparse
import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.init as init
from torchvision import datasets, transforms
from scatnet_learn.layers import InvariantLayerj1, InvariantLayerj1_dct
import torch.nn.functional as func
import numpy as np
import random

from ray.tune import Trainable

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('outdir', type=str, help='experiment directory')
parser.add_argument('--seed', type=int, default=None, metavar='S',
                    help='random seed (default: None)')
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--smoke-test', action="store_true",
                    help="Finish quickly for testing")


def net_init(m, gain=1):
    classname = m.__class__.__name__
    if (classname.find('Conv') != -1) or (classname.find('Linear') != -1):
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        try:
            init.constant_(m.bias, 0)
        # Can get an attribute error if no bias to learn
        except AttributeError:
            pass
    elif classname.find('InvariantLayerj1_dct') != -1:
        init.xavier_uniform_(m.A1, gain=gain)
        init.xavier_uniform_(m.A2, gain=gain)
        init.xavier_uniform_(m.A3, gain=gain)
    elif classname.find('InvariantLayerj1') != -1:
        init.xavier_uniform_(m.A, gain=gain)


class InvNet_shift(nn.Module):
    def __init__(self, C1=7, C2=49, shift='random'):
        super().__init__()
        if shift == 'dct':
            self.conv1 = InvariantLayerj1_dct(1, C1)
            self.conv2 = InvariantLayerj1_dct(C1, C2)
        else:
            self.conv1 = InvariantLayerj1(1, C1, alpha=shift)
            self.conv2 = InvariantLayerj1(C1, C2, alpha=shift)

        # Create the projection layer that doesn't need learning
        self.fc1 = nn.Linear(7*7*C2, 10)
        self.fc1.weight.requires_grad = False
        self.fc1.bias.requires_grad = False
        self.fc1.bias.data.zero_()
        self.fc2 = nn.Linear(10, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.shape[0], -1)
        y = func.relu(self.fc1(x))
        y = self.fc2(y)
        return func.log_softmax(y, dim=1)


class InvNet(nn.Module):
    def __init__(self, C1=7, C2=49, k=1):
        super().__init__()
        self.conv1 = InvariantLayerj1(1, C1, k=k)
        self.conv2 = InvariantLayerj1(C1, C2, k=k)

        # Create the projection layer that doesn't need learning
        self.fc1 = nn.Linear(7*7*C2, 10)
        self.fc1.weight.requires_grad = False
        self.fc1.bias.requires_grad = False
        self.fc1.bias.data.zero_()
        self.fc2 = nn.Linear(10, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.shape[0], -1)
        y = func.relu(self.fc1(x))
        y = self.fc2(y)
        return func.log_softmax(y, dim=1)


class ConvNet(nn.Module):
    def __init__(self, C1=7, C2=49, k=3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, C1, k, 1, padding=(k-1)//2)
        self.conv2 = nn.Conv2d(C1, C2, k, 1, padding=(k-1)//2)
        self.fc1 = nn.Linear(7*7*C2, 10)
        self.fc1.weight.requires_grad = False
        self.fc1.bias.requires_grad = False
        self.fc2 = nn.Linear(10, 10)

    def forward(self, x):
        x = func.relu(self.conv1(x))
        x = func.max_pool2d(x, 2, 2)
        x = func.relu(self.conv2(x))
        x = func.max_pool2d(x, 2, 2)
        x = x.view(x.shape[0], -1)
        y = func.relu(self.fc1(x))
        y = self.fc2(y)
        return func.log_softmax(y, dim=1)


class TrainMNIST(Trainable):
    """ This class handles model training and scheduling for our mnist networks.

    The config dictionary setup in the main function defines how to build the
    network. Then the experiment handler calles _train and _test to evaluate
    networks one epoch at a time.

    If you want to call this without using the experiment, simply ensure
    config is a dictionary with keys::

        - args: The parser arguments
        - type: The network type, one of 'conv', 'conv_wide', 'inv1x1',
            'inv3x3', 'inv_random', 'inv_dct', 'inv_impulse', 'inv_smooth'
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

        kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                '~/data',
                train=True,
                download=False,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307, ), (0.3081, ))
                ])),
            batch_size=args.batch_size,
            shuffle=True,
            **kwargs)
        self.test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                '~/data',
                train=False,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307, ), (0.3081, ))
                ])),
            batch_size=100,
            shuffle=True,
            **kwargs)

        # Build the network based on the type parameter. θ are the optimal
        # hyperparameters found by cross validation.
        C1 = 7
        C2 = 49
        if type_ == 'conv':
            self.model = ConvNet(C1, C2)
            θ = (0.1, 0.5, 1e-5, 1)
        elif type_ == 'conv_wide':
            C1 = 10
            C2 = 100
            self.model = ConvNet(C1, C2, k=5)
            θ = (0.1, 0.5, 1e-5, 1)
        elif type_ == 'inv1x1':
            self.model = InvNet(C1, C2, k=1)
            θ = (0.032, 0.9, 1e-4, 1)
        elif type_ == 'inv_impulse':
            self.model = InvNet_shift(C1, C2, shift='impulse')
            θ = (0.32, 0.5, 1e-4, 1)
        elif type_ == 'inv_smooth':
            self.model = InvNet_shift(C1, C2, shift='smooth')
            θ = (1.0, 0.0, 1e-5, 1)
        elif type_ == 'inv_random':
            self.model = InvNet_shift(C1, C2, shift='random')
            θ = (1.0, 0.9, 1e-5, 1)
        elif type_ == 'inv3x3':
            self.model = InvNet(C1, C2, k=3)
            θ = (0.1, 0.5, 1e-4, 1)
        elif type_ == 'inv_dct':
            self.model = InvNet_shift(C1, C2, shift='dct')
            θ = (1.0, 0, 1e-5, 1)
        else:
            raise ValueError('Unknown type')

        lr, mom, wd, std = θ
        # If the parameters were provided as an option, use them
        lr = config.get('lr', lr)
        mom = config.get('mom', mom)
        wd = config.get('wd', wd)
        std = config.get('std', std)
        init = lambda x: net_init(x, std)
        self.model.apply(init)
        self.model.cuda()

        self.optimizer = optim.SGD(
            self.model.parameters(), lr=lr, momentum=mom, weight_decay=wd)
        self.args = args

    def _train_iteration(self):
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = func.nll_loss(output, target)
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
        max_t=80,
        grace_period=20)

    tune.run_experiments(
        {
            exp_name: {
                "stop": {
                    #  "mean_accuracy": 0.95,
                    "training_iteration": 1 if args.smoke_test else 20,
                },
                "resources_per_trial": {
                    "cpu": 1,
                    "gpu": 0.3,
                },
                "run": TrainMNIST,
                #  "num_samples": 1 if args.smoke_test else 40,
                "num_samples": 10,
                "checkpoint_at_end": True,
                "config": {
                    "args": args,
                    "type": tune.grid_search([
                        'conv', 'inv1x1', 'inv3x3', 'inv_dct', 'inv_impulse',
                        'inv_smooth', 'inv_random', 'conv_wide']),
                    #  "type": tune.grid_search(['conv_wide']),
                    #  "lr": tune.grid_search([0.01, 0.0316, 0.1, 0.316, 1]),
                    #  "mom": tune.grid_search([0, 0.5, 0.9]),
                    #  "wd": tune.grid_search([1e-5, 1e-4]),
                    #  "std": tune.grid_search([0.5, 1., 1.5, 2.0])
                }
            }
        },
        verbose=1,
        scheduler=sched)
