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

from const import KPTS_15, SMPL_KPTS_15
from data_utils import sample_projection_matrix, project, \
        random_scale, move_to_center


DATASET_DIR = 'dataset/'
PEOPLE3D_DIR = '/home/kristijan/phd/datasets/people3d/'

GEN_H = 600
GEN_W = 600
P3D_H = 480
P3D_W = 640


# TODO: Move to data_utils.py.
def to_origin(pose_3d):
    for kpt_idx in range(pose_3d.shape[0]):
        pose_3d[kpt_idx] -= pose_3d[0]
    return pose_3d


def process_txt(fpath):
    kpts = []
    with open(fpath) as pf:
        lines = [x[:-1] for x in pf.readlines()][1:]
    for line_idx, line in enumerate(lines):
        if line_idx + 1 in KPTS_15:
            kpts.append([float(x) for x in line.split(' ')])
    return kpts


def process_json(json_path):
    pose_2d = np.zeros((15, 3), dtype=np.float32)
    with open(json_path) as fjson:
        data = json.load(fjson)
    try:
        pose_2d_tmp = np.array(
                data['people'][0]['pose_keypoints_2d'],
                dtype=np.float32)
        pose_2d[:, 0] = pose_2d_tmp[::3][:15]
        pose_2d[:, 1] = pose_2d_tmp[1::3][:15]
        pose_2d[:, 2] = np.array(1, dtype=np.float32)
    except:
        pass
    return pose_2d


def prepare_openpose(rootdir, dataset_name, scale=1.0, downscale=1.0, 
        orient_x=False, orient_z=False, train_ratio=0.8):

    def get_gender(gt_pose_dir):
        with open(os.path.join(gt_pose_dir, 'params.json')) as fjson:
            params = json.load(fjson)
            return params['gender']

    train_X = []
    train_Y = []
    test_X  = []
    test_Y  = []

    gt_dir = os.path.join(rootdir, 'gt/')
    openpose_dir = os.path.join(rootdir, 'openpose/')
    subject_dirs = [x for x in os.listdir(gt_dir) if 'npy' not in x]
    # Used for train/test split.
    num_dirs_per_gender = len(subject_dirs) / 2
    max_dir_idx = int(train_ratio * num_dirs_per_gender)

    # TODO: Merge GT and OpenPose into same function.
    for subject_dirname in [x for x in os.listdir(openpose_dir) if 'npy' not in x]:
        subject_dir = os.path.join(openpose_dir, subject_dirname)
        gt_pose_dir = os.path.join(gt_dir, subject_dirname)
        print(subject_dir)
        for pose_name in [x for x in os.listdir(subject_dir) if 'npy' in x]:
            pose_path = os.path.join(subject_dir, pose_name)
            pose_2d = process_json(pose_path)
            pose_2d[:, :2] = random_scale(pose_2d[:, :2], scale, downscale)
            pose_2d = move_to_center(pose_2d)
            # TODO: Avoid this magic number.
            pose_2d[:, :2] /= (600. - 1)
            pose_2d = np.expand_dims(pose_2d, axis=0)
            if int(subject_dirname[-4:]) < max_dir_idx:
                train_X.append(pose_2d)
                train_Y.append(get_gender(gt_pose_dir))
            else:
                test_X.append(pose_2d)
                test_Y.append(get_gender(gt_pose_dir))

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


def prepare_3dpeople_gt(rootdir, dataset_name, openpose=False):

    def get_gender(subject_dirname):
        if 'woman' in subject_dirname:
            return 1
        else:
            return 0

    subdir = 'openpose' if openpose else 'skeleton'
    rootdir = os.path.join(rootdir, subdir)

    train_X, train_Y, test_X, test_Y = [], [], [], []
    for data_type in ['train', 'test']:
        data_dir = os.path.join(rootdir, data_type)
        for subject_dirname in [x for x in sorted(os.listdir(data_dir)) if 'txt' not in x]:
            subject_dir = os.path.join(data_dir, subject_dirname)
            print(subject_dir)
            for action_dirname in sorted(os.listdir(subject_dir)):
                action_dir = os.path.join(subject_dir, action_dirname)
                pose_dir = os.path.join(action_dir, 'camera01')
                for pose_name in sorted(os.listdir(pose_dir)):
                    pose_path = os.path.join(pose_dir, pose_name)
                    if openpose:
                        pose_2d = process_json(pose_path)
                    else:
                        kpts = process_txt(pose_path)
                        pose_2d = np.array([x[:3] for x in kpts], dtype=np.float32)
                    pose_2d[:, 0] /= P3D_H
                    pose_2d[:, 1] /= P3D_W
                    pose_2d = move_to_center(pose_2d)
                    pose_2d[:, 2] = np.ones(pose_2d.shape[0])
                    pose_2d = np.expand_dims(pose_2d, axis=0)

                    if data_type == 'train':
                        train_X.append(pose_2d)
                        # TODO: Update this to also support identity.
                        train_Y.append(get_gender(subject_dirname))
                    else:
                        test_X.append(pose_2d)
                        test_Y.append(get_gender(subject_dirname))

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


def prepare_gender(rootdir, dataset_name, scale=1.0, downscale=1.0, 
        orient_x=False, orient_z=False, train_ratio=0.8):

    def get_gender(subject_dir):
        with open(os.path.join(subject_dir, 'params.json')) as fjson:
            params = json.load(fjson)
            return params['gender']

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
            P = sample_projection_matrix(orient_x, orient_z)
            pose_2d = project(pose_3d, P)[SMPL_KPTS_15]
            pose_2d[:, :2] = random_scale(pose_2d[:, :2], scale, downscale)
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
    parser.add_argument('--openpose', dest='openpose', action='store_true',
            help='use OpenPose predictions (not GT poses)')
    parser.add_argument('--scale', type=float, default=1.,
            help='main scale factor (upper and lower bound)')
    parser.add_argument('--downscale', type=float, default=1.,
            help='downscale factor (additional)')
    parser.add_argument('--orient_x', dest='orient_x', action='store_true',
            help='random X-axis transformation of the pose')
    parser.add_argument('--orient_z', dest='orient_z', action='store_true',
            help='random Z-axis transformation of the pose')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = init_parser()
    # TODO: Merge gender and identity functions into one and call
    # preparation functions by dataset name string.
    if args.task == 'gender':
        if args.dataset == 'people3d':
            prepare_3dpeople_gt(PEOPLE3D_DIR, args.name, args.openpose)
        else:
            prepare_gender(f'../smplx-generator/data/{args.dataset}/', 
                    args.name, args.scale, args.downscale)
    else:
        prepare_identity(f'../smplx-generator/data/{args.dataset}/', 
                args.name)

