# Fergal Cotter
#

# Future modules
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import subprocess
from py3nvml.utils import get_free_gpus
import numpy as np
import time
import argparse

DATASET = ['cifar10', 'cifar100']
TRAIN_SIZES = [10000, 50000]
C = [80, 96, 128]
CONV = ['invariantj1']
WD = [1e-4]
Nlayers = [2]


parser = argparse.ArgumentParser(description='''
PyTorch CIFAR10/CIFAR100 Training with standard and wavelet based convolutional 
layers. Designed to run on a multi-gpu system, and will run each experiment on a 
free gpu, one after another until all gpus are taken. Can be run on a cpu, but 
will be slow (perhaps restrict to low dataset sizes). Needs py3nvml to query the 
GPUs. For a multiple GPU system, will spin up subprocesses and use each one to 
run an experiment. For a CPU only test, will run one after the other.\n

The output will have directory structure:

    exp_dir/<dataset>/<layer_type>/<trainset_size>/<run_number>/

In each experiment directory, you will see the saved source (not useful for 
this experiment), a file called stdout (can inspect to see the printouts), 
the best checkpoint parameters, and tensorboard logs.

It is recommended to use tensorboard to view the run results.
''', formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('--exp_dir', type=str,
                    default='/scratch/fbc23/data/icip19_final3/',
                    help='Output directory for the experiment')
parser.add_argument('--train_sizes', type=int, nargs='+',
                    default=[1000, 2000, 5000, 10000, 20000, 50000],
                    help='Size of train sets to run experiments on. Provide as ' 
                         'a list after this flag.')
parser.add_argument('--datasets', type=str, nargs='+',
                    default=['cifar100', 'cifar10'],
                    choices=['cifar100', 'cifar10'],
                    help='List of datasets to run experiments on.')
parser.add_argument('-N', type=int, default=1, help='Number of runs')
parser.add_argument('--cpu', action='store_true', help='Run only on cpu')


runs = [
    ('tiny_imagenet', 20000, 'A'),
    ('tiny_imagenet', 20000, 'B'),
    ('tiny_imagenet', 20000, 'C'),
    ('tiny_imagenet', 20000, 'D'),
    ('tiny_imagenet', 20000, 'E'),
    ('tiny_imagenet', 20000, 'F'),
    ('tiny_imagenet', 20000, 'G'),
    ('tiny_imagenet', 20000, 'H'),
    ('tiny_imagenet', 20000, 'I'),
    ('tiny_imagenet', 20000, 'J'),
    ('tiny_imagenet', 20000, 'K'),
    ('tiny_imagenet', 20000, 'L'),
    ('tiny_imagenet', 20000, 'M'),
    ('tiny_imagenet', 20000, 'N'),
]


def main(args):
    for d, ts, c in runs:
        lr = 0.8
        mom = 0.85
        outdir = os.path.join(args.exp_dir, d, str(ts), 'Z' + c)
        os.makedirs(outdir, exist_ok=True)
        stdout_file = open(os.path.join(outdir, 'stdout'), 'w')
        if d == 'tiny_imagenet':
            wd = 5e-5
            steps = ['25', '35', '45']
            epochs = '50'
            data_dir = '/scratch/share/Tiny_ImageNet'
            eval_period = '1'
        else:
            wd = 1e-4
            steps = ['60', '80', '100']
            epochs = '120'
            data_dir = '/scratch/share/cifar'
            eval_period = '2'
        gpus = [str(x) for x in range(5)]
        cmd = ['python', '../main.py', outdir,
               '--trainsize', str(ts),
               '--eval_period', eval_period,
               '--no_comment',
               '--optim', 'sgd',
               '--lr', str(lr),
               '--momentum', str(mom),
               '--wd', str(wd),
               '--steps', *steps,
               '--epochs', epochs,
               '--gpu_select', *gpus,
               '--exist_ok',
               '--dataset', d,
               '--data_dir', data_dir,
               '--type', c]
        print('Running {} {}, group {}'.format(
                d, ts, c))

        if args.cpu:
            subprocess.run(cmd, stdout=stdout_file)
        else:
            # Only look at the last 4 gpus for the moment
            num_gpus = np.array(get_free_gpus())[:5].sum()
            while num_gpus < 1:
                time.sleep(10)
                num_gpus = np.array(get_free_gpus())[:5].sum()
            subprocess.Popen(cmd, stdout=stdout_file)
            # Give the processes time to start
            time.sleep(20)

# def main(args):
#     i = [0,] * len(CONV)
#     print('-' * 100)
#     for ts in TRAIN_SIZES:
#         ts_str = '{}k'.format(ts//1000)
#         print('-' * 50)
#         print('Trainset size: {}'.format(ts_str))
#         print('-' * 50)
#         for d in DATASET:
#             print('-' * 50)
#             print('{} Runs'.format(d))
#             print('-' * 50)
#             for c in C:
#                 for j, conv in enumerate(CONV):
#                     if conv == 'invariantj1':
#                         lr = 0.8
#                         mom = 0.85
#                     else:
#                         lr = 0.1
#                         mom = 0.9
#                     for wd in WD:
#                         for l in Nlayers:
#                             outdir = os.path.join(args.exp_dir, d, str(ts),
#                                                   conv, str(c))
#                             os.makedirs(outdir, exist_ok=True)
#                             stdout_file = open(os.path.join(outdir, 'stdout'), 'w')
#                             if isinstance(l, list):
#                                 layers = [str(x) for x in l]
#                             else:
#                                 layers = [str(l)]
#                             gpus = [str(x) for x in range(4, 8)]
#                             cmd = ['python', '../main.py', outdir,
#                                    '--conv_layer', conv,
#                                    '--trainsize', str(ts),
#                                    '--eval_period', '2',
#                                    '--no_comment',
#                                    '--optim', 'sgd',
#                                    '--lr', str(lr),
#                                    '--momentum', str(mom),
#                                    '--wd', str(wd),
#                                    '--steps', '60', '80', '100',
#                                    '--epochs', '120',
#                                    '--layers_per_scale', *layers,
#                                    '--channels_per_scale', str(c),
#                                    '--exist_ok',
#                                    '--dataset', d,
#                                    '--gpu_select', *gpus]
#                             print('Running {} group {}, l:{},\t'
#                                   'c:{},\t,lr:{},\tm:{}'.format(
#                                     conv, i[j], l, c, lr, mom))
#                             i[j] += 1
#
#                             if args.cpu:
#                                 subprocess.run(cmd, stdout=stdout_file)
#                             else:
#                                 # Only look at the last 4 gpus for the moment
#                                 num_gpus = np.array(get_free_gpus())[4:].sum()
#                                 while num_gpus < 1:
#                                     time.sleep(10)
#                                     num_gpus = np.array(get_free_gpus())[4:].sum()
#                                 subprocess.Popen(cmd, stdout=stdout_file)
#                                 # Give the processes time to start
#                                 time.sleep(20)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
