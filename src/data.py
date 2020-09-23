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

    def __call__(self, sample, epoch=None):
        torch_sample = {
            'X': torch.from_numpy(sample['X']),
            'Y': torch.from_numpy(sample['Y'])
            }
        return torch_sample


class NumViewsReductionTransformation(object):

    def __init__(self, num_epochs, num_views):
        self.num_epochs = num_epochs
        self.num_views = num_views
        # The number of views decrease by 4.
        # NOTE: The number of views are divisible by 4.
        self.epoch_step = num_epochs / (num_views / 4)

        self.prev = -1

    def __call__(self, sample, epoch):
        epoch_steps = int(epoch / self.epoch_step)
        start_idx = self.num_views - epoch_steps * 4
        if epoch_steps > self.prev:
            self.prev = epoch_steps
            print(epoch_steps)
            print(start_idx)
        sample['X'][:, :, start_idx:] *= 0.

        return sample


class ThreeDPeopleDataset(Dataset):

    def __init__(self, num_views=4, num_kpts=17, transforms=None, data_type=None, 
            mode=None):
        self.num_views = num_views
        self.num_kpts = num_kpts
        self.transforms = transforms
        self.data_type = data_type

        self.epoch = 0

        self.Y = np.load('./dataset/people3d/gt_{}.npy'.format(data_type))
        self.num_samples = self.Y.shape[0]

        print('>>> loading {} data'.format(data_type))

        if mode == 'openpose':
            print('>>> using openpose')
            self.X = np.load('./dataset/people3d/openpose_{}.npy'.format(data_type))
        elif mode == 'gt':
            self.X = np.load('./dataset/people3d/gt_2d_{}.npy'.format(data_type))
        elif mode == 'project':
            self.bar = Bar('>>>', fill='>', max=self.num_samples)
            self.Ps = generate_uniform_projection_matrices(self.num_views)
            self.X = self._generate_projections()

        self.X = np.swapaxes(self.X, 1, 3)
        

    def _generate_projections(self):
        X = np.empty((self.num_samples, self.num_views, self.num_kpts, 3), 
                dtype=np.float32)
        
        for data_idx in range(self.num_samples):
            self.bar.suffix = '({batch}/{size}) | Total: {ttl} | ETA: {eta:}' \
                .format(batch=data_idx + 1,
                        size=self.num_samples,
                        ttl=self.bar.elapsed_td,
                        eta=self.bar.eta_td)

            # Actual logic.
            x = []
            for view_idx in range(self.num_views):
                x.append(project(self.Y[data_idx], self.Ps[view_idx]))
            x = np.array(x, dtype=np.float32)
            X[data_idx] = x

            self.bar.next()

        self.bar.finish()

        return X

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sample = {'X': self.X[idx], 'Y': self.Y[idx].flatten()}
        if self.transforms:
            for transform in self.transforms:
                transform(sample, self.epoch)
        return sample


class H36MDataset():

    def __init__(self, num_views=4, num_kpts=15, transforms=None,
            data_type=None, mode=None):

        self.Y = np.load('./dataset/h36m/gt_{}.npy'.format(data_type))
        self.num_samples = self.Y.shape[0]
        self.num_views = num_views
        self.num_kpts = num_kpts

        if mode == 'openpose':
            self.X = np.load('./dataset/h36m/openpose_{}.npy'.format(data_type))
            self.X = np.swapaxes(self.X, 1, 3)
        elif mode == 'gt':
            self.X = np.load('./dataset/h36m/gt_2d_{}.npy'.format(data_type))
            self.X = np.swapaxes(self.X, 1, 3)
        else:
            self.Ps = generate_uniform_projection_matrices(self.num_views)
            self.X = self._generate_projections()

        self.transforms = transforms

        self.epoch = 0

    def _generate_projections(self):
        X = np.empty((self.num_samples, 3, self.num_kpts, self.num_views), 
                dtype=np.float32)

        for data_idx in range(self.num_samples):
            x = []
            for view_idx in range(self.num_views):
                x.append(project(self.Y[data_idx], self.Ps[view_idx]))
            x = np.array(x, dtype=np.float32)
            x = np.swapaxes(x, 0, 2)
            self.X[data_idx] = x

        return X

    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, idx):
        sample = {'X': self.X[idx], 'Y': self.Y[idx].flatten()}
        if self.transforms:
            for transform in self.transforms:
                transform(sample, self.epoch)
        return sample

