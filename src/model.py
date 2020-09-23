from __future__ import absolute_import
from __future__ import print_function

import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from os.path import join


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal(m.weight)


class Linear(nn.Module):
    def __init__(self, linear_size, p_dropout=0.5):
        super(Linear, self).__init__()
        self.l_size = linear_size

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)

        self.w2 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.w2(y)
        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = x + y

        return out


class LinearModel(nn.Module):
    def __init__(self,
                 linear_size=1024,
                 num_stage=2,
                 num_views=4,
                 num_kpts=15,
                 p_dropout=0.5,
                 test=False,
                 exp_dir=None):
        super(LinearModel, self).__init__()

        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage
        self.num_views = num_views
        self.num_kpts = num_kpts
        self.test = test
        self.weights_saved = False
        self.exp_dir = exp_dir

        # [2D] input_size: NUM_KPTS x NUM_COORD x NUM_VIEWS
        self.input_size = num_kpts * 2 * num_views
        # [3D] output_size: NUM_KPTS x NUM_COORD
        self.output_size = num_kpts * 3

        # process input to linear size
        self.w1 = nn.Linear(self.input_size, self.linear_size)
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)

        self.linear_stages = []
        for _ in range(num_stage):
            self.linear_stages.append(Linear(self.linear_size, self.p_dropout))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        # post processing
        self.w2 = nn.Linear(self.linear_size, self.output_size)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

    def forward(self, x):
        # pre-processing
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        # linear layers
        for i in range(self.num_stage):
            y = self.linear_stages[i](y)

        y = self.w2(y)

        if self.test and not self.weights_saved:
            self.weights_saved = True
            weights = self.w1.weight.detach().cpu().numpy()
            img_path = join(self.exp_dir, 'weights.png')
            np_path = join(self.exp_dir, 'weights.npy')
            plt.imsave(img_path, weights)
            np.save(np_path, weights)

        return y
