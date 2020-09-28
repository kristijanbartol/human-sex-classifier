import os
import numpy as np
from copy import deepcopy
from scipy.spatial.transform import Rotation as rot
import torch, torchvision
from time import time
import math
import h5py
import json
import random
import argparse

from const import SMPL_KPTS_15
from data_utils import generate_uniform_projection_matrices, project, \
        random_scale


DATASET_DIR = 'dataset/'


def to_origin(pose_3d):
    for kpt_idx in range(pose_3d.shape[0]):
        pose_3d[kpt_idx] -= pose_3d[0]
    return pose_3d


def prepare_identity(rootdir, dataset_name, scale=True, train_ratio=0.8):
    print(f'Dataset name: {dataset_name}')

    P = generate_uniform_projection_matrices(1)[0]
    train_X = []
    train_Y = []
    test_X  = []
    test_Y  = []

    gt_dir = os.path.join(rootdir, 'gt/')
    for subject_idx, subject_dirname in \
            enumerate([x for x in sorted(os.listdir(gt_dir)) if 'npy' not in x]):
        subject_dir = os.path.join(gt_dir, subject_dirname)
        print(subject_dirname)
        pose_dirs = [x for x in sorted(os.listdir(subject_dir)) if 'npy' in x]
        ratio_idx = int(train_ratio * len(pose_dirs))
        for pose_idx, pose_name in enumerate(pose_dirs):
            pose_path = os.path.join(subject_dir, pose_name)
            # TODO: Move 3D poses to the origin.
            pose_3d = np.load(pose_path)
            pose_3d = to_origin(pose_3d)
            pose_2d = project(pose_3d, P)[SMPL_KPTS_15]
            if scale:
                pose_2d = random_scale(pose_2d, var=0.5)
            # TODO: Normalize without a magic number 600.
            pose_2d[:, :2] /= (600. - 1)
            pose_2d = np.expand_dims(pose_2d, axis=0)

            if pose_idx > subject_idx and pose_idx < ratio_idx + subject_idx:
                train_X.append(pose_2d)
                train_Y.append(subject_idx)
            else:
                test_X.append(pose_2d)
                test_Y.append(subject_idx)

    prepared_dir = os.path.join(DATASET_DIR, dataset_name)
    os.makedirs(prepared_dir, exist_ok=True)
    train_X = np.array(train_X, dtype=np.float32)
    train_Y = np.array(train_Y, dtype=np.long)
    test_X  = np.array(test_X,  dtype=np.float32)
    test_Y  = np.array(test_Y,  dtype=np.long)
    np.save(os.path.join(prepared_dir, 'train_X.npy'), train_X)
    np.save(os.path.join(prepared_dir, 'train_Y.npy'), train_Y)
    np.save(os.path.join(prepared_dir, 'test_X.npy'), test_X)
    np.save(os.path.join(prepared_dir, 'test_Y.npy'), test_Y)


def prepare_gender(rootdir, dataset_name, scale=True, train_ratio=0.8):

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
    subject_dirs = [x for x in os.listdir(gt_dir) if 'npy' not in x]
    num_dirs_per_gender = len(subject_dirs) / 2
    max_dir_idx = int(train_ratio * num_dirs_per_gender)

    for subject_dirname in [x for x in os.listdir(gt_dir) if 'npy' not in x]:
        subject_dir = os.path.join(gt_dir, subject_dirname)
        print(subject_dir)
        for pose_name in [x for x in os.listdir(subject_dir) if 'npy' in x]:
            pose_path = os.path.join(subject_dir, pose_name)
            # TODO: Move 3D poses to the origin.
            pose_3d = np.load(pose_path)
            pose_2d = project(pose_3d, P)[SMPL_KPTS_15]
            if scale:
                pose_2d = random_scale(pose_2d, var=0.5)
            # TODO: Avoid this magic number.
            pose_2d[:, :2] /= (600. - 1)
            pose_2d = np.expand_dims(pose_2d, axis=0)
            if int(subject_dirname[-4:]) < max_dir_idx:
                train_X.append(pose_2d)
                train_Y.append(get_gender(subject_dir))
            else:
                test_X.append(pose_2d)
                test_Y.append(get_gender(subject_dir))

    prepared_dir = os.path.join(DATASET_DIR, dataset_name)
    os.makedirs(prepared_dir, exist_ok=True)

    train_X = np.array(train_X, dtype=np.float32)
    train_Y = np.array(train_Y, dtype=np.long)
    test_X  = np.array(test_X,  dtype=np.float32)
    test_Y  = np.array(test_Y,  dtype=np.long)
    np.save(os.path.join(prepared_dir, 'train_X.npy'), train_X)
    np.save(os.path.join(prepared_dir, 'train_Y.npy'), train_Y)
    np.save(os.path.join(prepared_dir, 'test_X.npy'), test_X)
    np.save(os.path.join(prepared_dir, 'test_Y.npy'), test_Y)


def init_parser():
    parser = argparse.ArgumentParser(
            description='Prepare datasets for learning.')
    parser.add_argument('--task', type=str,
            help='which task to prepare (gender/identity)')
    parser.add_argument('--dataset', type=str,
            help='which dataset (directory) to use')
    parser.add_argument('--name', type=str,
            help='name of a prepared dataset (directory)')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = init_parser()
    if args.task == 'gender':
        prepare_gender(f'../smplx-generator/data/{args.dataset}/', 
                args.name)
    else:
        prepare_identity(f'../smplx-generator/data/{args.dataset}/', 
                args.name)

