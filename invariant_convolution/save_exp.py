# Fergal Cotter
#

# Future modules
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import time
import sys
import git
from shutil import copyfile
import invariant_convolution as ic


TEMPLATE = """ICIP Experiment
===============

This experiment was run on {day} at {time}.
The command used to run the program was:

.. code:: 

    {runcmd}
    
The repo commit used at running this command was::

    {githash}
    
The numpy/pytorch random seed was::

    {seed}
    
The number of learnable parameters is::

    {num_params}

Description
-----------
"""

ACC_TEMPLATE = """
Best Acc
--------
The best validation accuracy was {acc:.2f}%
"""

def get_githash(module):
    try:
        git_repo = git.Repo(module.__file__, search_parent_directories=True)
        hash = str(git_repo.git.rev_parse('HEAD'))
    except git.InvalidGitRepositoryError:
        hash = "?"
    return hash


def break_run_cmd(params):
    cmd = 'python {file} {}'.format(' \n      '.join(
        params[1:]), file=params[0])
    return cmd


def get_num_params(net):
    n = 0
    if n is None:
        return '?'
    else:
        for p in net.parameters():
            if p.requires_grad:
                n += p.numel()

        if n < 1e5:
            s = '{:.2f}k'.format(n/1e3)
        elif n < 1e6:
            s = '{:.3f}M'.format(n/1e6)
        elif n < 1e7:
            s = '{:.2f}M'.format(n/1e6)
        else:
            s = '{:.1f}M'.format(n/1e6)
        return s


def save_experiment_info(outdir, seed, no_comment=False,
                         exist_ok=False, net=None):
    """ Creates an experiment info file in the output directory

    Args:
        outdir: the output directory
        net: the network object

    Returns:
        None
    """
    if os.path.exists(outdir):
        if 'debug' not in outdir and not exist_ok:
            raise ValueError('Output directory already exists')
    else:
        os.mkdir(outdir)

    file = os.path.join(outdir, 'INFO.rst')
    with open(file, 'w') as f:
        f.write(TEMPLATE.format(
            day=time.strftime('%Y/%m/%d'),
            time=time.strftime("%H-%M-%S", time.gmtime(time.time())),
            runcmd='python {}'.format(' '.join(sys.argv)),
            githash=get_githash(ic),
            seed=seed,
            num_params=get_num_params(net)
        ))
    if 'debug' not in outdir and not no_comment:
        os.system('vim + {file}'.format(file=file))

    print('Saved info file. Copying source')
    main = os.path.join(os.path.dirname(__file__), '..', 'main.py')
    learn = os.path.join(os.path.dirname(__file__), 'learn.py')
    netf = os.path.join(os.path.dirname(__file__), 'networks.py')
    layerf = os.path.join(os.path.dirname(__file__), 'layers.py')
    copyfile(main, os.path.join(outdir, 'main.py'))
    copyfile(learn, os.path.join(outdir, 'learn.py'))
    copyfile(netf, os.path.join(outdir, 'net.py'))
    copyfile(layerf, os.path.join(outdir, 'layers.py'))


def save_acc(outdir, acc):
    """ Append the best accuracy to the info file"""
    file = os.path.join(outdir, 'INFO.rst')
    if os.path.exists(file):
        with open(file, 'a') as f:
            f.write(ACC_TEMPLATE.format(acc=acc))

