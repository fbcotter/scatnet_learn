import os
import hashlib
from six.moves import urllib
import numpy as np
import sys


def download(url, dest_directory):
    filename = url.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)

    print("Download URL: {}".format(url))
    print("Download DIR: {}".format(dest_directory))

    def _progress(count, block_size, total_size):
        prog = float(count * block_size) / float(total_size) * 100.0
        sys.stdout.write('\r>> Downloading %s %.1f%%' %
                         (filename, prog))
        sys.stdout.flush()

    filepath, _ = urllib.request.urlretrieve(url, filepath,
                                             reporthook=_progress)
    print()
    return filepath


def md5(fname):
    hash_md5 = hashlib.md5()
    try:
        with open(fname, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        file_hash = hash_md5.hexdigest()
    except:
        file_hash = ''

    return file_hash


def convert_to_one_hot(vector, num_classes=None):
    """ Convert a one-of-k representation to one-hot
    Converts an input 1-D array/list of integers into an output
    array/list of one-hot vectors, where an i'th input value
    of j will set a '1' in the i'th row, j'th column of the
    output array.
    Example:
        v = np.array((1, 0, 4))
        one_hot_v = convertToOneHot(v)
        print one_hot_v
        [[0 1 0 0 0]
         [1 0 0 0 0]
         [0 0 0 0 1]]
    Parameters
    ----------
    vector : ndarray(float) or list(float)
        Array or list containing the one-of-k representations. Entries should be
        in the range [0, num_classes-1].
    num_classes : int or None
        How many classes there are. If set to None, will make it one more than
        the max number in the vector input.
    Returns
    -------
    y : ndarray(float) or list(ndarray)
        The result. Will return a list of 1-D arrays if fed a list of ints, or a
        2d array if fed a 1d array.
    """
    ret_list = False
    if isinstance(vector, int):
        scalar = vector
        if num_classes is None:
            raise ValueError('Cannot convert a single number to one-hot' +
                             'without knowing the number of classes')
        else:
            assert num_classes > 0
            assert num_classes > scalar

        return np.zeros((num_classes,)).itemset(scalar, 1)
    else:
        if isinstance(vector, list):
            ret_list = True
            vector = np.array(vector)

        if num_classes is None:
            num_classes = np.max(vector) + 1
        else:
            assert num_classes > 0
            assert num_classes > np.max(vector)

        result = np.zeros((len(vector), num_classes), np.int32)
        result[np.arange(len(vector)), vector] = 1

        if ret_list:
            result = np.split(result, len(result))
            result = [np.squeeze(x) for x in result]
        return result
