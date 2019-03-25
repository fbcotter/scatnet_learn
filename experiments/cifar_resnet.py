"""
This script allows you to run a host of tests on the invariant layer and
slightly different variants of it on CIFAR.
"""
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision import datasets
from scatnet_learn.layers import InvariantLayerj1, InvariantLayerj1_compress
import time
import sys
import torch.nn.functional as func
import numpy as np
import random
from scatnet_learn.data import cifar, tiny_imagenet
from scatnet_learn import optim

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
parser.add_argument('--depth', default=28, type=int, help='network depth')
parser.add_argument('--widen-factor', default=10, type=int,
                    help='multiple of 16 channels used for network')
parser.add_argument('--dropout', default=0.3, type=float,
                    help='Dropout rate')
parser.add_argument('--num-gpus', default=1, type=int)
parser.add_argument('--no-scheduler', action='store_true')
parser.add_argument('--ref', action='store_true')
# Core hyperparameters
parser.add_argument('--reg', default='l2', type=str, help='regularization term')
parser.add_argument('--steps', default=[60,80,100], type=int, nargs='+')
parser.add_argument('--gamma', default=0.2, type=float, help='Lr decay')


def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    return h, m, s


def num_correct(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k)
        return res, batch_size


def net_init(m):
    classname = m.__class__.__name__
    if (classname.find('Conv') != -1) or (classname.find('Linear') != -1):
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        try:
            init.constant_(m.bias, 0)
        # Can get an attribute error if no bias to learn
        except AttributeError:
            pass
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        try:
            init.constant_(m.bias, 0)
        # Can get an attribute error if no bias to learn
        except AttributeError:
            pass


class wide_basic(nn.Module):
    def __init__(self, C, F, p, stride=1, ref=False):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(C)
        self.conv1 = nn.Conv2d(C, F, 3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=p)
        self.bn2 = nn.BatchNorm2d(F)
        self.conv2 = nn.Conv2d(F, F, 3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or C != F:
            if ref:
                self.shortcut = nn.Conv2d(C, F, 1, stride=stride, bias=True)
            else:
                self.shortcut = InvariantLayerj1(C, F, stride, k=1,
                                                 alpha='impulse')

    def forward(self, x):
        out = self.dropout(self.conv1(func.relu(self.bn1(x))))
        out = self.conv2(func.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out


class Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, p, num_classes, ref=False):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16

        assert ((depth-4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)//6
        k = widen_factor

        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = nn.Conv2d(3, nStages[0], 3, 1, bias=True)
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, p, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, p, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, p, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, p, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, p, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def init(self, std=1):
        """ Define the default initialization scheme """
        self.apply(net_init)
        # Try do any custom layer initializations
        for child in self.children():
            try:
                child.init(std)
            except AttributeError:
                pass

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = func.relu(self.bn1(out))
        out = func.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return func.log_softmax(out, dim=-1)


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
            if args.dataset == 'cifar10':
                num_classes = 10
            elif args.dataset == 'cifar100':
                num_classes = 100
        elif args.dataset == 'tiny_imagenet':
            self.train_loader, self.test_loader = tiny_imagenet.get_data(
                64, args.data_dir, val_only=args.testOnly,
                batch_size=args.batch_size, trainsize=args.trainsize,
                seed=args.seed, distributed=False, **kwargs)
            num_classes = 200

        # ######################################################################
        # Build the network based on the type parameter. θ are the optimal
        # hyperparameters found by cross validation.
        θ = (0.05, 0.9, 5e-4, 1)

        lr, mom, wd, std = θ
        # If the parameters were provided as an option, use them
        lr = config.get('lr', lr)
        mom = config.get('mom', mom)
        wd = config.get('wd', wd)
        std = config.get('std', std)

        self.model = Wide_ResNet(args.depth, args.widen_factor, args.dropout,
                                 num_classes, ref=args.ref)
        self.model.init(std)
        if args.cuda:
            self.model.cuda()
            self.model = torch.nn.DataParallel(
                self.model, device_ids=range(torch.cuda.device_count()))

        # ######################################################################
        # Build the optimizer
        try:
            params = self.model.param_groups()
        except AttributeError:
            params = self.model.parameters()
        # Don't use the optimizer's weight decay, call that later in the loss
        # func
        self.optimizer, self.scheduler = optim.get_optim(
            'sgd', params, init_lr=lr, steps=args.steps, wd=wd,
            gamma=args.gamma, momentum=mom, max_epochs=args.epochs)
        self.args = args

    def _train_iteration(self):
        self.model.train()
        top1_correct = 0
        top5_correct = 0
        total = 0
        num_iter = len(self.train_loader)
        start = time.time()
        update_steps = np.linspace(
            int(1/4 * num_iter), num_iter-1, 4).astype('int')

        for batch_idx, (data, target) in enumerate(self.train_loader):
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = func.nll_loss(output, target)
            # Get the regularization loss directly from the network
            loss.backward()
            self.optimizer.step()

            # Plotting/Reporting
            if args.verbose:
                corrects, bs = num_correct(output.data, target, topk=(1,5))
                total += bs
                top1_correct += corrects[0]
                top5_correct += corrects[1]

                sys.stdout.write('\r')
                sys.stdout.write(
                    '| Epoch [{:3d}/{:3d}] Iter[{:3d}/{:3d}]\t\tLoss: {:.4f}\t'
                    'Acc@1: {:.3f}%\tAcc@5: {:.3f}%\tElapsed Time: '
                    '{:.1f}min'.format(
                        self.scheduler.last_epoch, 120, batch_idx+1, num_iter,
                        loss.item(), 100. * top1_correct.item()/total,
                        100. * top5_correct.item()/total, (time.time()-start)/60))
                sys.stdout.flush()
                if batch_idx in update_steps:
                    top1_correct = 0
                    top5_correct = 0
                    total = 0
                    print()

    def _test(self):
        self.model.eval()
        test_loss = 0
        top1_correct = 0
        top5_correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                if self.args.cuda:
                    data, target = data.cuda(), target.cuda()
                output = self.model(data)
                # sum up batch loss
                loss = func.nll_loss(output, target, reduction='sum')

                # get the index of the max log-probability
                corrects, bs = num_correct(output.data, target, topk=(1, 5))
                test_loss += loss.item()
                total += bs
                top1_correct += corrects[0]
                top5_correct += corrects[1]

        test_loss /= total
        acc1 = 100. * top1_correct.item()/total
        acc5 = 100. * top5_correct.item()/total
        if args.verbose:
            # Save checkpoint when best model
            sys.stdout.write('\r')
            print("\n| Validation Epoch #{}\t\t\tLoss: {:.4f}\tAcc@1: {:.2f}%\t"
                  "Acc@5: {:.2f}%".format(self.scheduler.last_epoch, test_loss,
                                          acc1, acc5))
        return {"mean_loss": test_loss, "mean_accuracy": acc1}

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
        cfg = {'args': args}
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

        m, b = linear_func(0.01, 0.93, 0.2, 0.85)
        tune.run_experiments(
            {
                exp_name: {
                    "stop": {
                        #  "mean_accuracy": 0.95,
                        "training_iteration": 1 if args.smoke_test else 120,
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
                    }
                }
            },
            verbose=1,
            scheduler=sched)
