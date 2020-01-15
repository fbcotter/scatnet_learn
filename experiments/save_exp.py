# Fergal Cotter
#

# Future modules
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import time
import git
import shutil


TEMPLATE = """Invariant Layer Experiment
==========================

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
Best Result
-----------
The best acc was {best:.3f} and the last acc was {last:.3f}
"""

FOLD_TEMPLATE = """
Fold {k}: {best:.3f}"""


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
    if net is None:
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


def save_experiment_info(outdir, seed, no_comment=False, net=None):
    """ Creates an experiment info file in the output directory

    Args:
        outdir: the output directory
        net: the network object

    Returns:
        None
    """
    file = os.path.join(outdir, 'INFO.rst')
    with open(file, 'w') as f:
        f.write(TEMPLATE.format(
            day=time.strftime('%Y/%m/%d'),
            time=time.strftime("%H-%M-%S", time.gmtime(time.time())),
            runcmd='python {}'.format(' '.join(sys.argv)),
            githash="?",
            seed=seed,
            num_params=get_num_params(net)
        ))
    if 'debug' not in outdir and not no_comment:
        os.system('vim + {file}'.format(file=file))

    print('Saved info file. Copying source')
    copytree(os.path.join(os.path.dirname(__file__), '..'), outdir)


def copytree(src, dst):
    """ Copies all the python files in src to dst recursively"""
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            if not os.path.isdir(d):
                os.mkdir(d)
            copytree(s, d)
        elif os.path.splitext(s)[1] == '.py':
            if not os.path.exists(d):
                shutil.copy2(s, d)


def save_acc(outdir, best, last):
    """ Append the best accuracy to the info file"""
    file = os.path.join(outdir, 'INFO.rst')
    if os.path.exists(file):
        with open(file, 'a') as f:
            f.write(ACC_TEMPLATE.format(best=best, last=last))


def save_kfoldacc(outdir, fold, r2):
    """ Append the best accuracy to the info file"""
    file = os.path.join(outdir, 'INFO.rst')
    if os.path.exists(file):
        with open(file, 'a') as f:
            if fold == 0:
                f.write("\nKFOLD Results\n-------------")
            f.write(FOLD_TEMPLATE.format(k=fold, best=r2))
