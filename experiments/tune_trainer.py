"""
This module creates a Ray Tune training class. In particular, it makes
scaffolding for running single epochs of training, testing, saving and loading
models. The Trainable class is then used in the experiment code with the
schedulers, but it can also be used without the scheduler.
"""

from ray.tune import Trainable
import time
import torch.nn.functional as func
import numpy as np
import torch
import sys
import os

import torch.nn.init as init


def net_init(m, gain=1):
    """ Function to initialize the networks. Needed by all experiments """
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


class BaseClass(Trainable):
    """ This class handles model training and scheduling for our networks.

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
        raise NotImplementedError("Please overwrite the _setup method")

    @property
    def verbose(self):
        return getattr(self, '_verbose', False)

    @property
    def use_cuda(self):
        if not hasattr(self, '_use_cuda'):
            self._use_cuda = torch.cuda.is_available()
        return self._use_cuda

    @property
    def last_epoch(self):
        if hasattr(self, 'scheduler'):
            return self.scheduler.last_epoch
        else:
            return self._last_epoch

    @property
    def final_epoch(self):
        if hasattr(self, '_final_epoch'):
            return self._final_epoch
        else:
            return 120

    def step_lr(self):
        if hasattr(self, 'scheduler'):
            self.scheduler.step()
        if hasattr(self, 'scheduler1'):
            self.scheduler1.step()

    def zero_grad(self):
        if hasattr(self, 'optimizer'):
            self.optimizer.zero_grad()
        if hasattr(self, 'optimizer1'):
            self.optimizer1.zero_grad()

    def opt_step(self):
        if hasattr(self, 'optimizer'):
            self.optimizer.step()
        if hasattr(self, 'optimizer1'):
            self.optimizer1.step()

    def _train_iteration(self):
        self.model.train()
        top1_update = 0
        top1_epoch = 0
        top5_update = 0
        top5_epoch = 0
        loss_update = 0
        loss_epoch = 0
        update = 0
        epoch = 0
        num_iter = len(self.train_loader)
        start = time.time()
        update_steps = np.linspace(
            int(1/4 * num_iter), num_iter-1, 4).astype('int')

        for batch_idx, (data, target) in enumerate(self.train_loader):
            if self.use_cuda:
                data, target = data.cuda(), target.cuda()
            self.zero_grad()
            output = self.model(data)
            loss = func.nll_loss(output, target)
            loss.backward()
            self.opt_step()
            corrects, bs = num_correct(output.data, target, topk=(1,5))
            top1_epoch += corrects[0].item()
            top5_epoch += corrects[1].item()
            loss_epoch += loss.item()*bs
            epoch += bs

            # Plotting/Reporting
            if self.verbose:
                update += bs
                top1_update += corrects[0].item()
                top5_update += corrects[1].item()
                loss_update += loss.item()*bs

                sys.stdout.write('\r')
                sys.stdout.write(
                    '| Epoch [{:3d}/{:3d}] Iter[{:3d}/{:3d}]\t\tLoss: {:.4f}\t'
                    'Acc@1: {:.3f}%\tAcc@5: {:.3f}%\tElapsed Time: '
                    '{:.1f}min'.format(
                        self.last_epoch, self.final_epoch, batch_idx+1,
                        num_iter, loss_update/update, 100. * top1_update/update,
                        100. * top5_update/update, (time.time()-start)/60))
                sys.stdout.flush()
                # Every update_steps, print a new line
                if batch_idx in update_steps:
                    top1_update = 0
                    top5_update = 0
                    loss_update = 0
                    update = 0
                    print()
        loss_epoch /= epoch
        top1_epoch = 100. * top1_epoch/epoch
        top5_epoch = 100. * top5_epoch/epoch
        return {"mean_loss": loss_epoch, "mean_accuracy": top1_epoch, "acc5":
                top5_epoch}

    def _test(self):
        self.model.eval()
        test_loss = 0
        top1_correct = 0
        top5_correct = 0
        epoch = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                if self.use_cuda:
                    data, target = data.cuda(), target.cuda()
                output = self.model(data)
                # sum up batch loss
                loss = func.nll_loss(output, target, reduction='sum')

                # get the index of the max log-probability
                corrects, bs = num_correct(output.data, target, topk=(1, 5))
                test_loss += loss.item()
                top1_correct += corrects[0].item()
                top5_correct += corrects[1].item()
                epoch += bs

        test_loss /= epoch
        acc1 = 100. * top1_correct/epoch
        acc5 = 100. * top5_correct/epoch
        if self.verbose:
            # Save checkpoint when best model
            print("|\n| Validation Epoch #{}\t\t\tLoss: {:.4f}\tAcc@1: {:.2f}%"
                  "\tAcc@5: {:.2f}%".format(
                      self.last_epoch, test_loss, acc1, acc5))
        return {"mean_loss": test_loss, "mean_accuracy": acc1, "acc5": acc5}

    def _train(self):
        if not hasattr(self, '_last_epoch'):
            self._last_epoch = 0
        else:
            self._last_epoch += 1
        self.step_lr()
        self._train_iteration()
        return self._test()

    def _save(self, checkpoint_dir, name='model.pth'):
        checkpoint_path = os.path.join(checkpoint_dir, name)
        if hasattr(self, 'optimizer1'):
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'optimizer1_state_dict': self.optimizer1.state_dict()
            }, checkpoint_path)
        else:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
            }, checkpoint_path)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if hasattr(self, 'optimizer1'):
            self.optimizer1.load_state_dict(checkpoint['optimizer1_state_dict'])
