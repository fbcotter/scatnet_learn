import torch.optim
from numpy import ndarray


def get_optim(optim, params, init_lr, steps=1, wd=0, gamma=1,
              momentum=0.9, max_epochs=120):
    if optim == 'sgd':
        optimizer = torch.optim.SGD(
            params, lr=init_lr, momentum=momentum, weight_decay=wd)
    elif optim == 'sgd_nomem':
        optimizer = torch.optim.SGD(
            params, lr=init_lr, momentum=0, weight_decay=wd)
    elif optim == 'adam':
        optimizer = torch.optim.Adam(
            params, lr=init_lr, weight_decay=wd,  # amsgrad=True,
            betas=(0.9, .999))
    else:
        raise ValueError('Unknown optimizer')

    # Set the learning rate decay
    if isinstance(steps, (tuple, list, ndarray)) and len(steps) == 1:
        steps = steps[0]

    if isinstance(steps, int):
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, int(max_epochs/steps), gamma=gamma)
    elif isinstance(steps, (tuple, list, ndarray)):
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, steps, gamma=gamma)
    else:
        raise ValueError('Unknown lr schedule')

    return optimizer, scheduler
