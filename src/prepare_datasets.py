import os
import numpy as np
from copy import deepcopy
#import cv2
from scipy.spatial.transform import Rotation as rot
import torch, torchvision
from time import time
import math
import h5py
import json

from const import H, W, PELVIS, RADIUS, K, H36M_WRITE_ROOT, H36M_READ_ROOT, \
        H36M_TRAIN, H36M_TEST, H36M_PELVIS, H36M_KPTS_15, KPTS_15
from data_utils import generate_uniform_projection_matrices, project


# TODO: Move this to data_utils.
def move_to_origin(data_3d):
    for pose_idx in range(data_3d.shape[0]):
        data_3d[pose_idx] -= data_3d[pose_idx][0]
    return data_3d


def prepare_gender(rootdir, train_ratio=0.8):

    def get_gender(subject_dir):
        with open(os.path.join(subject_dir, 'params.json')) as fjson:
            params = json.load(fjson)
            return params['gender']

    P = generate_uniform_projection_matrices(1)[0]
    train_X = []
    train_Y = []
    test_X  = []
    test_Y  = []

    gt_dir = os.path.join(rootdir, 'gt/')
    subject_dirs = [x for x os.listdir(gt_dir) if 'male' in x]
    num_dirs_per_gender = len(subject_dirs) / 2
    max_dir_idx = int(train_ratio * num_dirs_per_gender)

    for subject_dirname in os.listdir(gt_dir):
        subject_dir = os.path.join(gt_dir, subject_dirname)
        y = [0., 0.]
        y[get_gender(subject_dir)] = 1.
        for pose_name in [x for x in os.listdir(subject_dir) if 'npy' in x]:
            pose_path = os.path.join(subject_dir, pose_name)
            pose_3d = np.load(pose_path)
            pose_2d = project(pose_3d, P)
            if int(subject_dirname[-4:]) < max_dir_idx:
                train_X.append(pose_2d)
                train_Y.append(y)
            else:
                test_X.append(pose_2d)
                test_Y.append(y)

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)
    np.save('data/gender/X.npy', X)
    np.save('data/gender/Y.npy', Y)


if __name__ == '__main__':
    prepare_gender('../smplx-generator/data/gender/')

