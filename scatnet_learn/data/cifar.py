"""
Module to load in CIFAR10/100
"""
import torchvision
from torchvision import transforms
import numpy as np
import torch
import os
import random
import tarfile
import pickle
from scatnet_learn.utils import download, md5, convert_to_one_hot

mean = {
    'cifar10': (0.4914, 0.4822, 0.4465),
    'cifar100': (0.5071, 0.4867, 0.4408),
}

std = {
    'cifar10': (0.2023, 0.1994, 0.2010),
    'cifar100': (0.2675, 0.2565, 0.2761),
}

# Cifar folder names
CIFAR10_FOLDER = 'cifar-10-batches-py'
CIFAR100_FOLDER = 'cifar-100-python'

CIFAR10_URL_PYTHON = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
CIFAR10_MD5 = 'c58f30108f718f92721af3b95e74349a'
CIFAR100_URL_PYTHON = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
CIFAR100_MD5 = 'eb9058c3a382ffc7106e4002c42a8d85'


def _download_cifar(data_dir, cifar10=True):
    os.makedirs(data_dir, exist_ok=True)

    if cifar10:
        filename = CIFAR10_URL_PYTHON.split('/')[-1]
        data_cifar10 = os.path.join(data_dir, filename)

        # Don't re-download if it already exists
        if not os.path.exists(data_cifar10):
            need = True
        elif md5(data_cifar10) != CIFAR10_MD5:
            need = True
            print('File found but md5 checksum different. Redownloading.')
        else:
            print('Tar File found in dest_dir. Not Downloading again')
            need = False

        if need:
            print("Downloading Python CIFAR10 Data.")
            download(CIFAR10_URL_PYTHON, data_dir)

        # Extract and prepare CIFAR10 DATA
        print("Extracting Python CIFAR10 data.")
        tarfile.open(data_cifar10, 'r:gz').extractall(data_dir)
        print('Files extracted')

    else:
        # Download CIFAR100 DATA PYTHON
        filename = CIFAR100_URL_PYTHON.split('/')[-1]
        data_cifar100 = os.path.join(data_dir, filename)

        # Don't re-download if it already exists
        if not os.path.exists(data_cifar100):
            need = True
        elif md5(data_cifar100) != CIFAR100_MD5:
            need = True
            print('File found but md5 checksum different. Redownloading.')
        else:
            print('Tar File found in dest_dir. Not Downloading again.')
            need = False

        if need:
            print("Downloading Python CIFAR100 Data.")
            download(CIFAR100_URL_PYTHON, data_dir)

        # Extract and prepare CIFAR100
        print("Extracting Python CIFAR100 data.")
        tarfile.open(data_cifar100, 'r:gz').extractall(data_dir)
        print('Files extracted')


def load_cifar_data(data_dir, cifar10=True, val_size=2000, one_hot=True,
                    download=False):
    """Load cifar10 or cifar100 data
    Parameters
    ----------
    data_dir : str
        Path to the folder with the cifar files in them. These should be the
        python files as downloaded from `cs.toronto`__
        __ https://www.cs.toronto.edu/~kriz/cifar.html
    cifar10 : bool
        True if cifar10, false if cifar100
    val_size : int
        Size of the validation set.
    one_hot : bool
        True to return one hot labels
    download : bool
        True if you don't have the data and want it to be downloaded for you.
    Returns
    -------
    trainx : ndarray
        Array containing training images. There will be 50000 - `val_size`
        images in this.
    trainy : ndarray
        Array containing training labels. These will be one hot if the one_hot
        parameter was true, otherwise the standard one of k.
    testx : ndarray
        Array containing test images. There will be 10000 test images in this.
    testy : ndarray
        Test labels
    valx: ndarray
        Array containing validation images. Will be None if val_size was 0.
    valy: ndarray
        Array containing validation labels. Will be None if val_size was 0.
    """
    # Download the data if requested
    if download:
        _download_cifar(data_dir, cifar10)

    # Set up the properties for each dataset
    if cifar10:
        if CIFAR10_FOLDER in os.listdir(data_dir) and \
           'data_batch_1' not in os.listdir(data_dir):
            # move the data directory down one
            data_dir = os.path.join(data_dir, CIFAR10_FOLDER)
        train_files = ['data_batch_'+str(x) for x in range(1,6)]
        train_files = [os.path.join(data_dir, f) for f in train_files]
        test_files = ['test_batch']
        test_files = [os.path.join(data_dir, f) for f in test_files]
        num_classes = 10
        label_func = lambda x: np.array(x['labels'], dtype='int32')
    else:
        if CIFAR100_FOLDER in os.listdir(data_dir) and \
           'train' not in os.listdir(data_dir):
            # move the data directory down one
            data_dir = os.path.join(data_dir, CIFAR100_FOLDER)
        train_files = ['train']
        train_files = [os.path.join(data_dir, f) for f in train_files]
        test_files = ['test']
        test_files = [os.path.join(data_dir, f) for f in test_files]
        num_classes = 100
        label_func = lambda x: np.array(x['fine_labels'], dtype='int32')

    # Load the data into memory
    def load_files(filenames):
        data = np.array([])
        labels = np.array([])
        for name in filenames:
            with open(name, 'rb') as f:
                mydict = pickle.load(f, encoding='latin1')

            # The labels have different names in the two datasets.
            newlabels = label_func(mydict)
            if data.size:
                data = np.vstack([data, mydict['data']])
                labels = np.hstack([labels, newlabels])
            else:
                data = mydict['data']
                labels = newlabels
        data = np.reshape(data, [-1, 3, 32, 32], order='C')
        if one_hot:
            labels = convert_to_one_hot(labels, num_classes=num_classes)
        return data, labels

    train_data, train_labels = load_files(train_files)
    test_data, test_labels = load_files(test_files)
    if val_size > 0:
        train_data, val_data = np.split(train_data,
                                        [train_data.shape[0]-val_size])
        train_labels, val_labels = np.split(train_labels,
                                            [train_labels.shape[0]-val_size])
    else:
        val_data = None
        val_labels = None

    return train_data, train_labels, test_data, test_labels, val_data, \
        val_labels


def subsample(cifar10=True, size=50000):
    """ Subsamples cifar10/cifar100 so the entire dataset has size <size> but
    with equal classes."""
    basedir = os.path.dirname(__file__)
    if cifar10:
        class_sz = size // 10
        idx = np.load(os.path.join(basedir, 'cifar10_idxs.npy'))
        return np.sort(idx[:, :class_sz].ravel())
    else:
        class_sz = size // 100
        idx = np.load(os.path.join(basedir, 'cifar100_idxs.npy'))
        return np.sort(idx[:, :class_sz].ravel())


def get_data(in_size, data_dir, dataset='cifar10', batch_size=128,
             trainsize=-1, seed=random.randint(0, 10000), perturb=True,
             double_size=False, pin_memory=True, num_workers=0):
    """ Provides a pytorch loader to load in cifar10/100
    Args:
        in_size (int): the input size - can be used to scale the spatial size
        data_dir (str): the directory where the data is stored
        dataset (str): 'cifar10' or 'cifar100'
        batch_size (int): batch size for train loader. the val loader batch
            size is always 100
        trainsize (int): size of the training set. can be used to subsample it
        seed (int): random seed for the loaders
        perturb (bool): whether to do data augmentation on the training set
        double_size (bool): whether to double the input size
    """
    if double_size:
        resize = transforms.Resize(in_size*2)
    else:
        resize = transforms.Resize(in_size*1)

    if perturb:
        transform_train = transforms.Compose([
            transforms.RandomCrop(in_size, padding=4),
            resize,
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean[dataset], std[dataset])
        ])
    else:
        transform_train = transforms.Compose([
            transforms.CenterCrop(in_size),
            resize,
            transforms.ToTensor(),
            transforms.Normalize(mean[dataset], std[dataset])
        ])

    transform_test = transforms.Compose([
        transforms.CenterCrop(in_size),
        resize,
        transforms.ToTensor(),
        transforms.Normalize(mean[dataset], std[dataset]),
    ])

    if dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=False,
            transform=transform_train)
        if trainsize > 0:
            idxs = subsample(False, trainsize)
            trainset = torch.utils.data.Subset(trainset, idxs)
        testset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=False,
            transform=transform_test)

    elif dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(
            root=data_dir, train=True, download=False,
            transform=transform_train)
        if trainsize > 0:
            idxs = subsample(False, trainsize)
            trainset = torch.utils.data.Subset(trainset, idxs)
        testset = torchvision.datasets.CIFAR100(
            root=data_dir, train=False, download=False,
            transform=transform_test)

    # Set the loader initializer seeds for reproducibility
    def worker_init_fn(id):
        import random
        import numpy as np
        random.seed(seed+id)
        np.random.seed(seed+id)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        worker_init_fn=worker_init_fn, pin_memory=pin_memory)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=num_workers,
        worker_init_fn=worker_init_fn, pin_memory=pin_memory)

    return trainloader, testloader
