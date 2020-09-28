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

from const import KPTS_17, KPTS_23, KPTS_15, H36M_KPTS_17, H36M_KPTS_15, NUM_KPTS
from data_utils import generate_random_projection, random_translate, \
        generate_uniform_projection_matrices, project

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        torch_sample = {
            'X': torch.from_numpy(sample['X']),
            'Y': torch.from_numpy(sample['Y'])
            }
        return torch_sample


class ClassificationDataset(Dataset):

    def __init__(self, num_kpts=15, transforms=None, dataset='identity', data_type=None):
        self.num_kpts = num_kpts
        self.transforms = transforms
        self.data_type = data_type

        self.Y = np.load(f'./dataset/{dataset}/{data_type}_Y.npy')
        self.X = np.load(f'./dataset/{dataset}/{data_type}_X.npy')
        self.X = np.swapaxes(self.X, 1, 3)

        self.num_samples = self.Y.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # TODO: Might not need flatten() here.
        sample = {'X': self.X[idx], 'Y': self.Y[idx].flatten()}
        if self.transforms:
            for transform in self.transforms:
                transform(sample)
        return sample

