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

    def __init__(self, name, num_kpts, transforms, split, img_paths=None):
        self.name = name
        self.num_kpts = num_kpts
        self.transforms = transforms
        self.split = split
        self.rootdir = f'./dataset/{name}/'
        # TODO
        self.img_paths = img_paths
        
        print(f'>>> loading {name}/{split} dataset')

        self.Y = np.load(os.path.join(self.rootdir, f'{split}_Y.npy'))

        # TODO: Add different keypoints sets for different datasets.
        kpts_set = KPTS_DICT[num_kpts]
        self.X = np.load(os.path.join(self.rootdir, f'{split}_X.npy'))
        self.X = self.X[:, :, kpts_set, :]
        self.X = np.swapaxes(self.X, 1, 3)

        self.num_samples = self.Y.shape[0]

    def create_subsets(self):
        if self.split != 'test':
            print('WARNING: Do not create subsets if != test!')

        npy_files = [x for x in os.listdir(self.rootdir) \
                if 'train' not in x and 'test' not in x]
        subset_names = [x.split('_')[0] for x in npy_files]
        subsets = []
        for subset_name in np.unique(subset_names):
            subsets.append(ClassificationDataset(
                self.name, 
                self.num_kpts, 
                self.transforms,
                subset_name))

        if len(subsets) == 0:
            print('WARNING: Zero subsets, prepare the dataset!')

        return subsets

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sample = {'X': self.X[idx], 'Y': self.Y[idx].flatten()}
        if self.transforms:
            for transform in self.transforms:
                transform(sample)
        return sample


