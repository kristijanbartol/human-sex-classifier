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

from const import SMPL_KPTS_15
from data_utils import generate_uniform_projection_matrices, project


DATASET_DIR = 'dataset/'
GENDER_DIR = os.path.join(DATASET_DIR, 'gender/')
os.makedirs(GENDER_DIR, exist_ok=True)


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
    subject_dirs = [x for x in os.listdir(gt_dir) if 'male' in x]
    num_dirs_per_gender = len(subject_dirs) / 2
    max_dir_idx = int(train_ratio * num_dirs_per_gender)

    for subject_dirname in [x for x in os.listdir(gt_dir) if 'npy' not in x]:
        subject_dir = os.path.join(gt_dir, subject_dirname)
        print(subject_dir)
#        y = [0., 0.]
#        y[get_gender(subject_dir)] = 1.
        for pose_name in [x for x in os.listdir(subject_dir) if 'npy' in x]:
            pose_path = os.path.join(subject_dir, pose_name)
            pose_3d = np.load(pose_path)
            pose_2d = project(pose_3d, P)[SMPL_KPTS_15]
            pose_2d[:, :2] /= (600. - 1)
            pose_2d = np.expand_dims(pose_2d, axis=0)
            if int(subject_dirname[-4:]) < max_dir_idx:
                train_X.append(pose_2d)
#                train_Y.append(y)
                train_Y.append(get_gender(subject_dir))
            else:
                test_X.append(pose_2d)
#                test_Y.append(y)
                test_Y.append(get_gender(subject_dir))

    train_X = np.array(train_X, dtype=np.float32)
#    train_Y = np.array(train_Y, dtype=np.float32)
    train_Y = np.array(train_Y, dtype=np.long)
    test_X  = np.array(test_X,  dtype=np.float32)
#    test_Y  = np.array(test_Y,  dtype=np.float32)
    test_Y  = np.array(test_Y,  dtype=np.long)
    np.save(os.path.join(GENDER_DIR, 'train_X.npy'), train_X)
    np.save(os.path.join(GENDER_DIR, 'train_Y.npy'), train_Y)
    np.save(os.path.join(GENDER_DIR, 'test_X.npy'), test_X)
    np.save(os.path.join(GENDER_DIR, 'test_Y.npy'), test_Y)


if __name__ == '__main__':
    prepare_gender('../smplx-generator/data/gender/')

