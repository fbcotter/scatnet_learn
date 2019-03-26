from ray.tune import Trainable
import time
import torch.nn.functional as func
import numpy as np
import torch
import sys
import os


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
        raise NotImplementedError("Please overwrite the _setup metho")

    @property
    def verbose(self):
        return getattr(self, '_verbose', False)

    @property
    def use_cuda(self):
        if not hasattr(self, '_use_cuda'):
            self._use_cuda = torch.cuda.is_available()
        return self._use_cuda

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
            if self.use_cuda:
                data, target = data.cuda(), target.cuda()
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = func.nll_loss(output, target)
            # Get the regularization loss directly from the network
            loss.backward()
            self.optimizer.step()

            # Plotting/Reporting
            if self.verbose:
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
                        100. * top5_correct.item()/total,
                        (time.time()-start)/60))
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
                if self.use_cuda:
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
        if self.verbose:
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
