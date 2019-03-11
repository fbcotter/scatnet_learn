""" This script prepares the tiny imagenet dataset to make it more
amenable to pytorch loading.

I.e. it puts the validation files in a new folder called 'val2' where
each class is in its own subfolder, rather than relying on the
val_annotations.txt file."""
import csv
import os
from shutil import copyfile
import argparse

parser = argparse.ArgumentParser(description='Prep tiny imagenet for '
                                             'classification')
parser.add_argument('data_dir', type=str,
                    default='/scratch/share/Tiny_Imagenet',
                    help='Default location for the dataset')


def main(args):
    labels = {}
    with open(os.path.join(args.data_dir, 'val', 'val_annotations.txt')) as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            labels[row[0]] = row[1]

    folders = {}
    for k, v in labels.items():
        try:
            folders[v].append(k)
        except KeyError:
            folders[v] = [k]

    # Make a new folder call val2
    os.makedirs(os.path.join(args.data_dir, 'val2'), exist_ok=True)
    for k, v in folders.items():
        os.makedirs(os.path.join(args.data_dir, 'val2', k), exist_ok=True)
        for f in v:
            copyfile(os.path.join(args.data_dir, 'val', 'images', f),
                     os.path.join(args.data_dir, 'val2', k, f))


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
