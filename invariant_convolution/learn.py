"""
Module to do generic training/validating of a pytorch network. On top of the
loss function, will report back accuracy.
"""
# Future modules
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import torch
import torch.utils.data
import torch.autograd as autograd
import numpy as np
import time


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


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calculate_plot_steps(num_iter, pts):
    startpoint = 1/pts * (num_iter-1)
    update_steps = np.linspace(startpoint, num_iter-1, pts, endpoint=True)
    return update_steps.astype('int')


def get_lr(optim):
    lrs = []
    for p in optim.param_groups:
        lrs.append(p['lr'])
    if len(lrs) == 1:
        return lrs[0]
    else:
        return lrs


def train(loader, net, loss_fn, optimizer, epoch=0, epochs=0,
          use_cuda=True, writer=None, summary_freq=4):
    """ Train a model with the given loss functions

    Args:
        loader: pytorch data loader. needs to spit out a triple of
            (x, target) where target is an int representing the class number.
        net: nn.Module that spits out a prediction
        loss_fn: Loss function to apply to model output. Loss function should
            accept 3 inputs - (output, months, target).
        optimizer: any pytorch optimizer
        epoch (int): current epoch
        epochs (int): max epochs
        use_cuda (bool): true if want to use gpu
        writer: tensorboard writer
        summary_freq: number of times to update the
    """
    net.train()
    train_loss = 0
    top1_correct = 0
    top5_correct = 0
    total = 0

    losses = AverageMeter()
    num_iter = len(loader)
    update_steps = np.linspace(int(1/summary_freq * num_iter),
                               num_iter-1,
                               summary_freq).astype('int')
    start = time.time()

    print('\n=> Training Epoch #%d, LR=%.4f' % (epoch, get_lr(optimizer)))
    with autograd.detect_anomaly():
        for batch_idx, (inputs, targets) in enumerate(loader):
            # GPU settings
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            # Forward and Backward
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_fn(outputs, targets)
            try:
                loss += net.get_reg()
            except AttributeError:
                try:
                    loss += net.module.get_reg()
                except AttributeError:
                    pass
            loss.backward()
            optimizer.step()

            # Plotting/Reporting
            train_loss += loss.item()
            losses.update(loss.item())
            corrects, bs = num_correct(outputs.data, targets, topk=(1,5))
            total += bs
            top1_correct += corrects[0]
            top5_correct += corrects[1]

            sys.stdout.write('\r')
            sys.stdout.write(
                '| Epoch [{:3d}/{:3d}] Iter[{:3d}/{:3d}]\t\tLoss: {:.4f}\t'
                'Acc@1: {:.3f}%\tAcc@5: {:.3f}%\tElapsed Time: '
                '{:.1f}min'.format(
                    epoch, epochs, batch_idx+1, num_iter, loss.item(),
                    100. * top1_correct.item()/total,
                    100. * top5_correct.item()/total, (time.time()-start)/60))
            sys.stdout.flush()

            # Output summaries
            if batch_idx in update_steps and writer is not None:
                global_step = 100*epoch + int(100*batch_idx/num_iter)
                writer.add_scalar('acc', 100. * top1_correct.item()/total,
                                  global_step)
                writer.add_scalar('acc5', 100. * top5_correct.item()/total,
                                  global_step)
                writer.add_scalar('loss', losses.avg, global_step)
                top1_correct = 0
                top5_correct = 0
                total = 0
                losses.reset()
                print()


def validate(loader, net, loss_fn=None, use_cuda=True, epoch=-1, writer=None,
             noise=None, insertlevel=0):
    """ Validate a model with the given loss functions

    Args:
        loader: pytorch data loader. needs to spit out a tuple of
            (x, target) where target is an integer representing the class of
            the input.
        net: nn.Module that spits out a prediction
        loss_fn: Loss function to apply to model output. Can be none and loss
            reporting won't be done for validation steps.
        use_cuda (bool): if true, will put things on the gpu
        epoch: current epoch (used only for print and logging purposes)
        writer: tensorboard writer
        noise: None or std of noise to add to input

    Returns:
        acc: current epoch accuracy
    """
    net.eval()
    test_loss = 0
    top1_correct = 0
    top5_correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            # Calculate the output (potentially with noise)
            if noise is not None:
                outputs = net.forward_noise(inputs, insertlevel, noise)
            else:
                outputs = net(inputs)

            if loss_fn is not None:
                loss = loss_fn(outputs, targets)
            else:
                loss = torch.tensor(0)

            corrects, bs = num_correct(outputs.data, targets, topk=(1, 5))
            test_loss += bs * loss.item()
            total += bs
            top1_correct += corrects[0]
            top5_correct += corrects[1]

    # Save checkpoint when best model
    test_loss /= total
    acc1 = 100. * top1_correct.item()/total
    acc5 = 100. * top5_correct.item()/total
    sys.stdout.write('\r')
    print("\n| Validation Epoch #{}\t\t\tLoss: {:.4f}\tAcc@1: {:.2f}%\t"
          "Acc@5: {:.2f}%".format(epoch, test_loss, acc1, acc5))

    if writer is not None and epoch >= 0:
        writer.add_scalar('loss', test_loss, 100*epoch)
        writer.add_scalar('acc', acc1, 100*epoch)
        writer.add_scalar('acc5', acc5, 100*epoch)

    return acc1, acc5
