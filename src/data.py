from __future__ import print_function, division
import os
import math
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torchvision import transforms, utils
from os.path import join, basename, normpath, exists, dirname
from progress.bar import Bar as Bar
from time import time
import json

from const import SMPL_KPTS_SUB_15, SMPL_KPTS_SUB_11, SMPL_KPTS_SUB_10, \
        SMPL_KPTS_SUB_9, SMPL_KPTS_SUB_5, SMPL_KPTS_SUB_4, SMPL_KPTS_SUB_2

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


KPTS_DICT = {
        15: SMPL_KPTS_SUB_15,
        11: SMPL_KPTS_SUB_11,
        10: SMPL_KPTS_SUB_10,
        9 : SMPL_KPTS_SUB_9,
        5 : SMPL_KPTS_SUB_5,
        4 : SMPL_KPTS_SUB_4,
        2 : SMPL_KPTS_SUB_2
}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        torch_sample = {
            'X': torch.from_numpy(sample['X']),
            'Y': torch.from_numpy(sample['Y'])
            }
        return torch_sample


class ClassificationDataset(Dataset):

    def __init__(self, name, num_kpts, transforms, split,
            arch='cnn', gt=False, gt_paths=None, 
            img_paths=None):
        self.name = name
        self.num_kpts = num_kpts
        self.transforms = transforms
        self.split = split
        self.arch = arch
        self.gt = gt
        self.rootdir = f'./dataset/{name}/'
        self.gt_paths = gt_paths
        self.img_paths = img_paths

        if self.gt:
            # For GT, be able to select keypoint subsets.
            kpts_set = KPTS_DICT[num_kpts]
        else:
            # For OpenPose, use first 15 keypoints.
            kpts_set = range(15)

        print(f'>>> loading {name}/{split} dataset')

        self.Y = np.load(os.path.join(self.rootdir, f'{split}_Y.npy'))
        self.X = np.load(os.path.join(self.rootdir, f'{split}_X.npy'))
        self.X = self.X[:, :, kpts_set, :]
        if arch == 'cnn':
            self.X = np.swapaxes(self.X, 1, 3)
        else:
            self.X = np.squeeze(self.X, axis=1)
            self.X = self.X[:, :, :2]
            self.X = np.reshape(self.X, (-1, 30))

    def __load_paths(self, subset_name, path_type):
        paths_path = os.path.join(self.rootdir, 
                f'{subset_name}_{path_type}paths.txt')

        if os.path.exists(paths_path):
            with open(paths_path) as path_f:
                paths = [x[:-1] for x in path_f.readlines()]
        else:
            paths = None
            print(f'>>> NOTE: {paths_path} not found. This is'
                    'expected for PETA test dataset')

        return paths

    def create_subsets(self):
        if self.split == 'train':
            print('WARNING: Do not create subsets on train set!')

        npy_files = [x for x in os.listdir(self.rootdir) \
                if 'train' not in x and 'test' not in x and 'valid' not in x \
                and 'npy' in x]

        subset_names = [x[:-6] for x in npy_files]
        subsets = []

        for subset_name in np.unique(subset_names):
            gt_paths = self.__load_paths(subset_name, 'gt')
            img_paths = self.__load_paths(subset_name, 'img')
            subsets.append(ClassificationDataset(
                self.name, 
                self.num_kpts, 
                self.transforms,
                split=subset_name,
                arch=self.arch,
                gt=self.gt,
                gt_paths=gt_paths,
                img_paths=img_paths))

        if len(subsets) == 0:
            print('WARNING: Zero subsets, prepare the dataset!')

        return subsets

    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, idx):
        sample = {'X': self.X[idx], 'Y': self.Y[idx].flatten()}
        if self.transforms:
            for transform in self.transforms:
                transform(sample)
        return sample

