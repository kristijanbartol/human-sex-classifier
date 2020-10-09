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


kpts_dict = {
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

    def __init__(self, name, num_kpts, transforms, data_type):
        self.num_kpts = num_kpts
        self.transforms = transforms
        self.data_type = data_type
        self.rootdir = f'./dataset/{name}/'
        
        print(f'>>> loading {name} ({data_type}) dataset')

        self.Y = np.load(os.path.join(self.rootdir, f'{data_type}_Y.npy'))

        # TODO: Add different keypoints sets for different datasets.
        kpts_set = kpts_dict[num_kpts]
        self.X = np.load(os.path.join(self.rootdir, f'{data_type}_X.npy'))
        self.X = self.X[:, :, kpts_set, :]
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

    def report(self):
        if self.data_type != 'test':
            raise Exception('Trying to report on train set!')


class People3D(ClassificationDataset):

    def __init__(self, name, num_kpts, transforms, data_type):
        super().__init__(name, num_kpts, transforms, data_type)
        if data_type == 'test':
            dict_template = os.path.join(
                    self.rootdir, 'test_{}_idxs_dict.json')
            with open(dict_template.format('action')) as fjson:
                self.action_idxs_dict = json.load(fjson)
            with open(dict_template.format('subject')) as fjson:
                self.subject_idxs_dict = json.load(fjson)

    def __report(self, scores, dict_type):
        report_dict = {}
        if dict_type == 'action':
            dict_ = self.action_idxs_dict
        else:
            dict_ = self.subject_idxs_dict
        for action in self.action_idxs_dict:
            error = 0.
            correct_counter = 0
            for idx in self.action_idxs_dict[action]:
                error += np.abs(scores[idx][1] - scores[idx][2])
                correct_counter += scores[idx][0]
            num_samples = len(self.action_idxs_dict[action])
            mean_error = error / num_samples
            accuracy = float(correct_counter) / num_samples
            action_report_dict[action] = {
                'error': mean_error,
                'accuracy': accuracy
                # TODO: Add the rest of the confusion matrix.
            }
        with open(os.path.join(self.rootdir, 
            f'report_per_{dict_type}.json', 'w')) as fjson:
            json.dump(report_dict, fjson)

    def __extract_top(self, order, num_top=100):
        reverse = False if 'best' else True
        scores = sorted(scores, key=lambda x: np.abs(x[1] - x[2]), 
                reverse=reverse)
        top_samples = scores[:num_top]
        with open(os.path.join(self.rootdir,
            f'report_{order}_{num_top}.json'), 'w') as f:
            for sample_idx in top_samples:
                f.write(str(sample_idx) + '\n')

    def report(self, scores):
        super().report()
        self.__report(scores, 'action')
        self.__report(scores, 'subject')
        self.__extract_top(scores, 'best')
        self.__extract_top(scores, 'worst')


