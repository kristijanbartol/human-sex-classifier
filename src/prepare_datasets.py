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


# TODO: Move this to data_utils.
def move_to_origin(data_3d):
    for pose_idx in range(data_3d.shape[0]):
        data_3d[pose_idx] -= data_3d[pose_idx][0]
    return data_3d


def generate_list(data_type):
    path_list = []
    subject_dirnames = sorted(os.listdir('/data/people3d/gt/'))
    if data_type == 'train':
        subject_dirnames = [x for x in subject_dirnames if int(x[-2:]) < 34]
    else:
        subject_dirnames = [x for x in subject_dirnames if int(x[-2:]) >= 34]

    for subject_dirname in subject_dirnames:
        print('{}: {}'.format(data_type, subject_dirname))
        subject_dir = os.path.join('/data/people3d/gt/', subject_dirname)
        for action_dirname in sorted(os.listdir(subject_dir)):
            action_dir = os.path.join(subject_dir, action_dirname)
            # TODO: Remove generated .npy files and update this line.
            camera_dirnames = [x for x in sorted(os.listdir(action_dir)) 
                    if 'camera' in x]
            camera0_dir = os.path.join(action_dir, camera_dirnames[0])

            for pose_fname in sorted(os.listdir(camera0_dir)):
                sample_list = [os.path.join(camera0_dir, pose_fname)]
                for camera_dirname in camera_dirnames[1:]:
                    camera_dir = os.path.join(action_dir, camera_dirname)
                    pose_path = os.path.join(camera_dir, pose_fname)
                    sample_list.append(pose_path)
                path_list.append(' '.join(sample_list))
    with open('./dataset/people3d/{}.txt'.format(data_type), 
            'w') as store_file:
        store_file.write('\n'.join(path_list) + '\n')


def prepare_openpose(data_type):
    with open('./dataset/people3d/{}.txt'.format(data_type)) as lfile:
        samples_list = [x[:-1] for x in lfile.readlines()]

    X = []
    non_zero_samples = []
    for sample_idx, sample_list in enumerate(samples_list):
        x = np.zeros((4, 15, 3), dtype=np.float32)
        counter = 0
        for view_idx, path_2d in enumerate(sample_list.split(' ')):
            path_2d = path_2d.replace('gt', 'openpose').replace(
                    '.txt', '_keypoints.json')
            try:
                with open(path_2d) as f:
                    data = json.load(f)
                    try:
                        pose_2d_tmp = np.array(data['people'][0]['pose_keypoints_2d'], 
                                dtype=np.float32)
                        pose_2d = np.zeros((15, 3))
                        pose_2d[:, 0] = pose_2d_tmp[::3][:15]
                        pose_2d[:, 1] = pose_2d_tmp[1::3][:15]
                        pose_2d[:, 2] = np.array(1, dtype=np.float32)
                    except:
                        pose_2d = np.zeros((15, 3))
                        counter += 1
                x[view_idx] = pose_2d
            except:
                print('Unable to open {} - need to regerate OpenPose!'.format(
                    path_2d))
                continue

        if counter != 4:
            X.append(x)
            non_zero_samples.append(samples_list[sample_idx])
        else:
            print('No person detected in any frame ({})...'.format(path_2d))

    X = np.array(X)
    np.save('./dataset/people3d/openpose_{}.npy'.format(data_type), X)
    with open('./dataset/people3d/{}.txt'.format(data_type), 
            'w') as store_file:
        store_file.write('\n'.join(non_zero_samples) + '\n')


def prepare_gt(data_type, openpose=False):

    def process_kpts(fpath):
        kpts = []
        try:
            with open(fpath) as kpt_f:
                lines = [x[:-1] for x in kpt_f.readlines()][1:]
        except:
            return None
        for line_idx, line in enumerate(lines):
            if line_idx + 1 in KPTS_15:
                kpts.append([float(x) for x in line.split(' ')])
        return kpts

    # NOTE: The samples' list file depends on whether or not OpenPose is ON.
    with open('./dataset/people3d/{}.txt'.format(data_type)) as lf:
        samples_list = [x[:-1] for x in lf.readlines()]
    X = np.empty((len(samples_list), 4, 15, 3), dtype=np.float32)
    Y = np.empty((len(samples_list), 15, 3), dtype=np.float32)

    for sample_idx, fpaths in enumerate(samples_list):
        print(fpaths.split(' ')[0])
        x = np.empty((4, 15, 3), dtype=np.float32)
        for cam_idx, fpath in enumerate(fpaths.split(' ')):
            kpts = process_kpts(fpath)
            # In case the file does not exist in cam != 0, just leave zeros.
            if kpts is None:
                continue
            x[cam_idx] = np.array([x[:3] for x in kpts], dtype=np.float32)
            # The third coordinate of 2D GT should be dummy ones.
            x[cam_idx, :, 2] = np.ones(15)
            if cam_idx == 0:
                y = np.array([x[3:] for x in kpts], dtype=np.float32)
                y = move_to_origin(y)
                y[:, [0,1,2]] = y[:, [0,2,1]]
        X[sample_idx, :, :] = x
        Y[sample_idx, :, :] = y

    np.save('./dataset/people3d/gt_2d_{}.npy'.format(data_type), X)
    np.save('./dataset/people3d/gt_{}.npy'.format(data_type), Y)


def process_openpose_sample(sample_flist):
    x = np.zeros((4, 15, 3), dtype=np.float32)
    no_kpts_counter = 0
    for view_idx, path_2d in enumerate(sample_flist):
        with open(path_2d) as f:
            data = json.load(f)
            try:
                pose_2d_tmp = np.array(
                        data['people'][0]['pose_keypoints_2d'], 
                        dtype=np.float32)
                pose_2d = np.zeros((15, 3))
                pose_2d[:, 0] = pose_2d_tmp[::3][:15]
                pose_2d[:, 1] = pose_2d_tmp[1::3][:15]
                pose_2d[:, 2] = np.array(1, dtype=np.float32)
            except:
                pose_2d = np.zeros((15, 3))
                no_kpts_counter += 1
        x[view_idx] = pose_2d

    if no_kpts_counter == 4:
        print('No person detected in any frame ({})...'.format(path_2d))
        return None
    else:
        return x


def process_action(action_dir, openpose=False):

    def check_sample(gt_pose_fname):
        sample_flist = []
        op_pose_fname = gt_pose_fname.replace(
                '.jpg', '_keypoints.json')
        for op_cam_dirname in op_cam_dirs:
            op_cam_dir = os.path.join(op_dir, op_cam_dirname)
            op_pose_path = os.path.join(op_cam_dir, op_pose_fname)

            if os.path.exists(op_pose_path):
                sample_flist.append(op_pose_path)
            else:
                print('Path {} does not exist!'.format(op_pose_path))
                return None
        return sample_flist

    h5_fpath = os.path.join(action_dir, 'annot.h5')

    gt_dir = os.path.join(action_dir, 'imageSequence')
    op_dir = action_dir.replace('processed', 'openpose')

    print(action_dir)

    gt_cam_dirs = [x for x in sorted(os.listdir(gt_dir))]
    op_cam_dirs = [x for x in sorted(os.listdir(op_dir)) if 'npy' not in x]

    gt_cam0_dir = os.path.join(gt_dir, gt_cam_dirs[0])
    op_cam0_dir = os.path.join(op_dir, op_cam_dirs[0])

    with h5py.File(h5_fpath, 'r') as h5_f:
        data_3d = h5_f['pose']['3d'][()][:, H36M_KPTS_15, :] / 100.
        tmp_2d = h5_f['pose']['2d'][()][:, H36M_KPTS_15, :]

    assert(tmp_2d.shape[0] == data_3d.shape[0])
    # NOTE: Do not try to do this properly, because:
    # - you need to distinguish whether OpenPose or GT is wrong,
    # - if GT is wrong, need to count how many samples, to remove them,
    # - they logic is already complicated for OpenPose itself.
    if data_3d.shape[0] % 4 != 0:
        print('Action {} missing some GT files - skipping!'.format(
            action_dir))
        return None

    num_samples = int(data_3d.shape[0] / 4)
    data_3d = data_3d[:num_samples]

    # Assert that the number of files in the folder matches H5 num_samples.
    assert(len(os.listdir(gt_cam0_dir)) == num_samples)

    data_op = []
    if openpose:
        not_ok_idxs = []
        for sample_idx, gt_pose_fname in \
                enumerate(sorted(os.listdir(gt_cam0_dir))):
            sample_ok = True
            sample_flist = check_sample(gt_pose_fname)
            if sample_flist is None:
                sample_ok = False
            else:
                x = process_openpose_sample(sample_flist)
                if x is None:
                    sample_ok = False

            if sample_ok:
                data_op.append(x)
            else:
                not_ok_idxs.append(sample_idx)

        data_3d = np.delete(data_3d, not_ok_idxs, axis=0)
        slices = []
        for not_ok_idx in not_ok_idxs:
            slices += list(range(not_ok_idx, num_samples * 4, num_samples))
        tmp_2d = np.delete(tmp_2d, slices, axis=0)
        data_op = np.array(data_op, dtype=np.float32)

    # Expand first axis into size (-1, 4).
    tmp_2d = tmp_2d.reshape((4, -1, tmp_2d.shape[1], tmp_2d.shape[2]))
    tmp_2d = np.swapaxes(tmp_2d, 0, 1)
    # Add ones on the 3rd dim of last axis.
    data_2d = np.ones((tmp_2d.shape[0], tmp_2d.shape[1], 
        tmp_2d.shape[2], tmp_2d.shape[3] + 1))
    data_2d[:, :, :, :-1] = tmp_2d

    return data_2d, move_to_origin(data_3d), data_op


def prepare_h36m(data_type, openpose=True):
    h36m_2d_data = []
    h36m_gt_data = []
    h36m_op_data = []
    for subject_idx in H36M_TRAIN if data_type == 'train' else H36M_TEST:
        subject_name = 'S{}'.format(subject_idx)
        subject_dir = os.path.join('/data/h36m/processed/', subject_name)
        for action_dirname in sorted(os.listdir(subject_dir)):
            action_dir = os.path.join(subject_dir, action_dirname)
            result = process_action(action_dir, openpose)
            if result is None:
                continue
            else:
                h36m_2d_data.append(result[0])
                h36m_gt_data.append(result[1])
                h36m_op_data.append(result[2])

    h36m_2d_data = np.concatenate(h36m_2d_data, axis=0).astype(np.float32)
    np.save('./dataset/h36m/gt_2d_{}.npy'.format(data_type), h36m_2d_data)
    assert(h36m_2d_data.shape[3] == 3)
    h36m_gt_data = np.concatenate(h36m_gt_data, axis=0).astype(np.float32)
    np.save('./dataset/h36m/gt_{}.npy'.format(data_type), h36m_gt_data)
    if openpose:
        h36m_op_data = np.concatenate(h36m_op_data, axis=0).astype(np.float32)
        np.save('./dataset/h36m/openpose_{}.npy'.format(data_type), h36m_op_data)
        assert(h36m_gt_data.shape[0] == h36m_op_data.shape[0])


def prepare_openpose_h36m(data_type):
    with open('./dataset/h36m/{}.txt'.format(data_type)) as lf:
        samples_list = [x[:-1] for x in lf.readlines()]

    X = []
    non_zero_samples = []
    for sample_idx, sample_list in enumerate(samples_list):
        x = np.zeros((4, 15, 3), dtype=np.float32)
        counter = 0
        for view_idx, path_2d in enumerate(sample_list.split(' ')):
            with open(path_2d) as f:
                data = json.load(f)
                try:
                    pose_2d_tmp = np.array(data['people'][0]['pose_keypoints_2d'], 
                            dtype=np.float32)
                    pose_2d = np.zeros((15, 3))
                    pose_2d[:, 0] = pose_2d_tmp[::3][:15]
                    pose_2d[:, 1] = pose_2d_tmp[1::3][:15]
                    pose_2d[:, 2] = np.array(1, dtype=np.float32)
                except:
                    pose_2d = np.zeros((15, 3))
                    counter += 1
            x[view_idx] = pose_2d

        if counter != 4:
            X.append(x)
            non_zero_samples.append(samples_list[sample_idx])
        else:
            print('No person detected in any frame ({})...'.format(path_2d))

    X = np.array(X)
    np.save('./dataset/h36m/openpose_{}.npy'.format(data_type), X)
    with open('./dataset/h36m/{}.txt'.format(data_type), 
            'w') as store_file:
        store_file.write('\n'.join(non_zero_samples) + '\n')


if __name__ == '__main__':
#    generate_list('train')
#    prepare_openpose('train')
    prepare_gt('train')
#    generate_list('test')
#    prepare_openpose('test')
    prepare_gt('test')

#    prepare_h36m('train', openpose=True)
#    prepare_h36m('test', openpose=True)

