import os
import sys
import cv2
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
        random_scale, move_to_center, move_from_center


DATASET_DIR = 'dataset/'
PEOPLE3D_DIR = '/home/kristijan/phd/datasets/people3d/'


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
#        pose_2d[:, 2] = np.array(1, dtype=np.float32)
        pose_2d[:, 2] = pose_2d_tmp[2::3][:15]

        missing_parts = 0
        conf = 0.
        for kpt_idx, kpt in enumerate(pose_2d):
            if not np.any(kpt[:2]):
                missing_parts += 1
            conf += kpt[2]
#        conf = np.sum(pose_2d_tmp[2::3][:15]) / 15
        conf /= 15
    except:
        return pose_2d, None, None

    return pose_2d, missing_parts, conf
    

# TODO: Move to data_utils.py
def set_missing_joints(pose_2d):
    # Mid hip.
    if not np.any(pose_2d[8][:2]):
        if not np.any(pose_2d[1][:2]):
            return None
        pose_2d[8] = pose_2d[1]

    # Right leg.
    idxs = [8, 9, 10, 11]
    for idx in idxs[1:]:
        if not np.any(pose_2d[idx][:2]):
            pose_2d[idx] = pose_2d[idx-1]

    # Left leg.
    idxs = [8, 12, 13, 14]
    for idx in idxs[1:]:
        if not np.any(pose_2d[idx][:2]):
            pose_2d[idxs] = pose_2d[idx-1]

    # Neck.
    if not np.any(pose_2d[1][:2]):
        pose_2d[1] = pose_2d[8]

    # Right arm.
    idxs = [1, 2, 3, 4]
    for idx in idxs[1:]:
        if not np.any(pose_2d[idx][:2]):
            pose_2d[idx] = pose_2d[idx-1]

    # Left arm.
    idxs = [1, 5, 6, 7]
    for idx in idxs[1:]:
        if not np.any(pose_2d[idx][:2]):
            pose_2d[idx] = pose_2d[idx-1]

    # Nose.
    if not np.any(pose_2d[0][:2]):
        pose_2d[0] = pose_2d[1]

    return pose_2d


def prepare_peta(rootdir, dataset_name, centered=False, 
        split_ratios=(0.8, 0.1, 0.1)):

    def get_gender_label_idx(line):
        for item_idx, item in enumerate(line.split(' ')):
            if item == 'personalMale' or item == 'personalFemale':
                return item_idx

    def get_img_size(img_path):
        img = cv2.imread(img_path)
        img_size = max(img.shape[0], img.shape[1])
        return img_size

    train_X, test_X, valid_X, train_Y, valid_Y, test_Y = [], [], [], [], [], []
    subdir_dict = {}
    openpose_dir = os.path.join(rootdir, 'openpose')
    invalid_counters, missing_parts_counter, avg_confs, male_counter, female_counter =\
            [], [], [], [], []
    last_idx = 0
    for xdir_idx, xdir_name in enumerate(sorted(os.listdir(openpose_dir))):
        invalid_counters.append(0)
        missing_parts_counter.append(0)
        male_counter.append(0)
        female_counter.append(0)
        avg_confs.append(0.)
        subdir_dict[xdir_name] = { 'X' : [], 'Y': [], 'img': [] }
        xdir = os.path.join(openpose_dir, xdir_name)
        img_dir = os.path.join(rootdir, 'imgs/', xdir_name, 'archive/')
        label_path = os.path.join(img_dir, 'Label.txt')
        fnames = sorted(os.listdir(xdir), key=lambda x: int(x.split('_')[0]))

        with open(label_path) as flabel:
            labels = [x[:-1] for x in flabel.readlines()]
        id_gender_dict = {}
        gender_label_idx = get_gender_label_idx(labels[0])
        for line in labels:
            id_ = int(line.split(' ')[0].split('.')[0])
            gender = None
            for item_idx, item in enumerate(line.split(' ')):
                if item == 'personalMale' or item == 'personalFemale':
                    break
            label = line.split(' ')[item_idx]
            if label == 'personalMale':
                gender = 0
            elif label == 'personalFemale':
                gender = 1
            id_gender_dict[id_] = gender
        
        img_ext = os.listdir(img_dir)[0].split('.')[1]
        img_names = [x.replace('_keypoints.json', f'.{img_ext}') \
                for x in fnames]
        img_paths = [os.path.join(img_dir, x) for x in img_names]


        train_valid_part = split_ratios[0] + split_ratios[1]
        train_part = split_ratios[0] / train_valid_part

        train_valid_size = train_valid_part * len(fnames)
        train_size = train_part * train_valid_size

        train_valid_idxs = np.random.choice(
                len(fnames), int(train_valid_size), replace=False)
        train_idxs = train_valid_idxs[:int(train_size)]
        valid_idxs = train_valid_idxs[int(train_size):]

        test_idxs = np.arange(len(fnames))
        test_idxs = np.delete(test_idxs, train_valid_idxs, axis=0)
        idxs_dict = { 'train': train_idxs, 
                      'valid': valid_idxs, 
                      'test': test_idxs }


        for idx_set_key in idxs_dict:
            for idx in idxs_dict[idx_set_key]:
                kpt_path = os.path.join(xdir, fnames[idx])
                pose_2d, num_missing, conf = process_json(kpt_path)
                img_size = get_img_size(img_paths[idx])
                pose_2d[:, :2] /= img_size

                if centered:
                    pose_2d = move_to_center(pose_2d)

                pose_2d = np.expand_dims(pose_2d, axis=0)
                subject_id = int(fnames[idx].split('.')[0].split('_')[0])
                gender = id_gender_dict[subject_id]

#                pose_2d[0] = set_missing_joints(pose_2d[0])
                if gender is not None and np.any(pose_2d):
#                if gender is not None and not np.all(np.isnan(pose_2d[0])):
                    missing_parts_counter[xdir_idx] += num_missing
                    avg_confs[xdir_idx] += conf
                    if gender == 0:
                        male_counter[xdir_idx] +=1
                    else:
                        female_counter[xdir_idx] += 1

                    if idx_set_key == 'train':
                        train_X.append(pose_2d)
                        train_Y.append(gender)
                    elif idx_set_key == 'valid':
                        valid_X.append(pose_2d)
                        valid_Y.append(gender)
                    else:
                        test_X.append(pose_2d)
                        test_Y.append(gender)
                        subdir_dict[xdir_name]['X'].append(pose_2d)
                        subdir_dict[xdir_name]['Y'].append(gender)
                        subdir_dict[xdir_name]['img'].append(img_paths[idx])
                else:
                    invalid_counters[xdir_idx] += 1
        
        num_ok = len(fnames) - invalid_counters[xdir_idx]
        avg_missing = missing_parts_counter[xdir_idx] / num_ok
        avg_conf = avg_confs[xdir_idx] / num_ok
        print(f'{xdir_name} ({num_ok}/{len(fnames)}): {avg_missing}, {avg_conf}: '
                f'm{male_counter[xdir_idx]}, f{female_counter[xdir_idx]}')

    prepared_dir = os.path.join(DATASET_DIR, dataset_name)
    os.makedirs(prepared_dir, exist_ok=True)

    train_X = np.array(train_X, dtype=np.float32)
    train_Y = np.array(train_Y, dtype=np.long)
    valid_X = np.array(valid_X, dtype=np.float32)
    valid_Y = np.array(valid_Y, dtype=np.long)
    test_X  = np.array(test_X, dtype=np.float32)
    test_Y  = np.array(test_Y, dtype=np.long)

    for subdir_key in subdir_dict:
        subdir_dict[subdir_key]['X'] = np.array(
                subdir_dict[subdir_key]['X'], dtype=np.float32)
        subdir_dict[subdir_key]['Y'] = np.array(
                subdir_dict[subdir_key]['Y'], dtype=np.long)
        np.save(os.path.join(prepared_dir, f'{subdir_key}_X.npy'), 
                subdir_dict[subdir_key]['X'])
        np.save(os.path.join(prepared_dir, f'{subdir_key}_Y.npy'), 
                subdir_dict[subdir_key]['Y'])
        with open(os.path.join(prepared_dir, 
            f'{subdir_key}_imgpaths.txt'), 'w') as f:
            for img_path in subdir_dict[subdir_key]['img']:
                f.write(img_path + '\n')

    np.save(os.path.join(prepared_dir, 'train_X.npy'), train_X)
    np.save(os.path.join(prepared_dir, 'train_Y.npy'), train_Y)
    np.save(os.path.join(prepared_dir, 'valid_X.npy'), valid_X)
    np.save(os.path.join(prepared_dir, 'valid_Y.npy'), valid_Y)
    np.save(os.path.join(prepared_dir, 'test_X.npy'), test_X)
    np.save(os.path.join(prepared_dir, 'test_Y.npy'), test_Y)

    num_invalid = sum(invalid_counters)
    total_samples = num_invalid + train_X.shape[0] + \
            valid_X.shape[0] + test_X.shape[0]
    print(f'Number of invalid samples: {num_invalid}')
    print(f'Total number of samples: {total_samples}')


def prepare_3dpeople(rootdir, dataset_name, openpose=False, centered=False):

    def get_gender(subject_dirname):
        if 'woman' in subject_dirname:
            return 1
        else:
            return 0

    # NOTE: The rootdir is either openpose or GT dir.
    # The GT and img dirs are used for analyses' purposes.
    subdir = 'openpose' if openpose else 'gt'
    rootdir = os.path.join(rootdir, subdir)

    train_X, train_Y, test_X, test_Y = [], [], [], []
    subdir_dict = {}

    for data_type in ['train', 'test']:
        data_dir = os.path.join(rootdir, data_type)
        for subject_dirname in [x for x in sorted(
                os.listdir(data_dir)) if 'txt' not in x]:

            if data_type == 'test':
                subdir_dict[subject_dirname] = { 
                        'X' : [], 'Y': [], 'img': [], 'gt': [] }

            subject_dir = os.path.join(data_dir, subject_dirname)
            print(subject_dir)
            for action_dirname in sorted(os.listdir(subject_dir)):

                action_dir = os.path.join(subject_dir, action_dirname)
                pose_dir = os.path.join(action_dir, 'camera01')

                if data_type == 'test':
                    if action_dirname not in subdir_dict:
                        subdir_dict[action_dirname] = { 
                                'X' : [], 'Y': [], 'img': [], 'gt': [] }

                    # NOTE: Replace 'openpose' or 'gt' with 'gt'.
                    gt_dir = pose_dir.replace(subdir, 'gt') 
                    img_dir = pose_dir.replace(subdir, 'imgs')
                    img_dir = os.path.join(img_dir, os.listdir(img_dir)[0])

                for pose_name in sorted(os.listdir(pose_dir)):
                    pose_path = os.path.join(pose_dir, pose_name)

                    if data_type == 'test':
                        img_path = os.path.join(img_dir, f'{pose_name[:4]}.jpg')
                        gt_path = os.path.join(gt_dir, f'{pose_name[:4]}.txt')

                        if not os.path.exists(img_path):
                            continue

                    if openpose:
                        pose_2d = process_json(pose_path)
                    else:
                        kpts = process_txt(pose_path)
                        pose_2d = np.array([x[:3] for x in kpts], 
                                dtype=np.float32)
                    pose_2d[:, :2] /= 640.
                    if centered:
                        pose_2d = move_to_center(pose_2d)
                    pose_2d[:, 2] = np.ones(pose_2d.shape[0])
                    pose_2d = np.expand_dims(pose_2d, axis=0)

                    gender = get_gender(subject_dirname)
                    if data_type == 'train':
                        train_X.append(pose_2d)
                        train_Y.append(gender)
                    else:
                        test_X.append(pose_2d)
                        test_Y.append(gender)

                        subdir_dict[action_dirname]['X'].append(
                                pose_2d)
                        subdir_dict[action_dirname]['Y'].append(
                                gender)
                        subdir_dict[action_dirname]['img'].append(
                                img_path)
                        subdir_dict[action_dirname]['gt'].append(
                                gt_path)
                        subdir_dict[subject_dirname]['X'].append(
                                pose_2d)
                        subdir_dict[subject_dirname]['Y'].append(
                                gender)
                        subdir_dict[subject_dirname]['img'].append(
                                img_path)
                        subdir_dict[subject_dirname]['gt'].append(
                                gt_path)

    prepared_dir = os.path.join(DATASET_DIR, dataset_name)
    os.makedirs(prepared_dir, exist_ok=True)
    train_X = np.array(train_X, dtype=np.float32)
    train_Y = np.array(train_Y, dtype=np.long)
    test_X  = np.array(test_X,  dtype=np.float32)
    test_Y  = np.array(test_Y,  dtype=np.long)
    
    for subdir_key in subdir_dict:
        subdir_dict[subdir_key]['X'] = np.array(
                subdir_dict[subdir_key]['X'], dtype=np.float32)
        subdir_dict[subdir_key]['Y'] = np.array(
                subdir_dict[subdir_key]['Y'], dtype=np.long)

        np.save(os.path.join(prepared_dir, f'{subdir_key}_X.npy'), 
                subdir_dict[subdir_key]['X'])
        np.save(os.path.join(prepared_dir, f'{subdir_key}_Y.npy'), 
                subdir_dict[subdir_key]['Y'])

        with open(os.path.join(prepared_dir, 
            f'{subdir_key}_imgpaths.txt'), 'w') as f:
            for img_path in subdir_dict[subdir_key]['img']:
                f.write(img_path + '\n')

        with open(os.path.join(prepared_dir,
            f'{subdir_key}_gtpaths.txt'), 'w') as f:
            for gt_path in subdir_dict[subdir_key]['gt']:
                f.write(gt_path + '\n')

    np.save(os.path.join(prepared_dir, 'train_X.npy'), train_X)
    np.save(os.path.join(prepared_dir, 'train_Y.npy'), train_Y)
    np.save(os.path.join(prepared_dir, 'test_X.npy'), test_X)
    np.save(os.path.join(prepared_dir, 'test_Y.npy'), test_Y)


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


def prepare_gender(rootdir, dataset_name, scale=1.0, downscale=1.0, 
        centered=False, orient_x=False, orient_z=False, train_ratio=0.8):

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
            if not centered:
                pose_2d = move_from_center(pose_2d)
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
    parser.add_argument('--dataset', type=str, 
            choices=['people3d', 'smpl', 'peta'],
            help='which dataset to use (not directory)')
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
    parser.add_argument('--centered', dest='centered', action='store_true',
            help='centered 2D input poses')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = init_parser()
    if args.dataset == 'people3d':
        prepare_3dpeople(PEOPLE3D_DIR, args.name, args.openpose, 
                args.centered)
    elif args.dataset == 'smpl':
        # TODO: Merge OpenPose and GT into the same function.
        if args.openpose:
            prepare_openpose(f'../smplx-generator/data/{args.dataset}',
                    args.name, args.scale, args.downscale, args.centered)
        else:
            prepare_gender(f'../smplx-generator/data/{args.dataset}/', 
                    args.name, args.scale, args.downscale, args.centered)
    else:
        prepare_peta('/home/kristijan/phd/datasets/PETA/', args.name)
#        prepare_peta('/data/PETA/', args.name)

