import os
import numpy as np
from copy import deepcopy
from scipy.spatial.transform import Rotation as rot
import torch, torchvision
from time import time
import math
import random
from sklearn import metrics

from const import H, W, PELVIS, RADIUS, K, KPTS_15


def one_hot(labels, num_classes):
    oh_labels = np.zeros((labels.size, num_classes))
    oh_labels[np.arange(labels.size), labels] = 1.
    return oh_labels


def calc_auc(targets, outputs):
    '''
    Calculate AUC.
    
    Need to first perform argsort per labels, as
    sklearn expects it.

    :targets: labels
    :outputs: model predictions

    :auc: area under curve for given predictions
    '''
    sort_idxs = np.argsort(targets)
    targets = targets[sort_idxs]
    outputs = outputs[sort_idxs]

    fpr, tpr, thresholds = metrics.roc_curve(
            targets, outputs, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    return auc


def load_pose_2d_txt(gt_path):
    kpts = []
    with open(gt_path) as gt_f:
        lines = [x[:-1] for x in gt_f.readlines()][1:]
    for line_idx, line in enumerate(lines):
        if line_idx + 1 in KPTS_15:
            kpts.append([float(x) for x in line.split(' ')][:2])
    return kpts


def load_gt(gt_paths):
    gt_X = []

    for gt_path in gt_paths:
        pose_2d = load_pose_2d_txt(gt_path)
        gt_X.append(pose_2d)

    gt_X = np.array(gt_X)
    return gt_X


def mpjpe_2d_openpose(est, gt):
    '''
    Calculate MPJPE for non-zero keypoint estimations.
    '''
    assert(est.shape[0] == gt.shape[0])

    est = deepcopy(est)
    est = est[:, :2, :, :]
    est *= 640.
    est = np.swapaxes(est, 1, 3)

    mmpjpe = 0.
    for pose_idx in range(est.shape[0]):
        mpjpe = 0.
        counter = 0

        for kpt_idx in range(est.shape[2]):
            kpt = est[pose_idx, 0, kpt_idx, :2]
            gt_ = gt[pose_idx, kpt_idx]
            if np.any(kpt):
                mpjpe += np.linalg.norm(kpt - gt_)
                counter += 1

        if counter > 0:
            mmpjpe += mpjpe / counter
        else:
            # Some 2D poses are all zeros.
            continue

    mmpjpe = (mmpjpe / est.shape[0]) / 640. * 100.
    return mmpjpe


def mean_missing_parts(est):
    '''
    Calculate mean number of missing (OpenPose) parts.
    '''
    num_missing = 0

    est = deepcopy(est)
    est = est[:, :2, :, :]
    est = np.swapaxes(est, 1, 3)

    for pose_2d in est:
        for kpt in pose_2d[0]:
            if not np.any(kpt):
                num_missing += 1

    return float(num_missing) / est.shape[0]


def random_scale(pose_2d, scale=1.0, downscale=1.0):
    '''
    Random scale while keeping the pose location.
    '''
    scale_factor = np.random.uniform(
            1. / (scale * downscale), scale, 1)
    max_coord = np.amax(pose_2d, axis=0)
    min_coord = np.amin(pose_2d, axis=0)
    mid_point = (max_coord + min_coord) / 2.
    pose_2d -= mid_point
    pose_2d *= scale_factor
    pose_2d += mid_point
    return pose_2d


def translate_2d(pose_2d, t):
    for kpt_idx in range(pose_2d.shape[0]):
        # Apply only to non-zero keypoints.
        if np.any(pose_2d[kpt_idx]):
            pose_2d[kpt_idx] += t / 2.
    return pose_2d


def move_2d_to_origin(pose_2d):
    max_coord = np.amax(pose_2d, axis=0)
    min_coord = np.amin(pose_2d, axis=0)
    mid_point = (max_coord + min_coord) / 2.

    pose_2d = translate_2d(pose_2d, -mid_point)
    return pose_2d


def fit_to_frame(pose_2d, frame_size):
    '''
    Fit to visualization frame of given size.

    :pose_2d: pose kpts without hom coordinate
    '''
    pose_2d = move_2d_to_origin(pose_2d)
    pose_2d *= frame_size
    pose_2d = translate_2d(
            pose_2d, float(frame_size) / 2.)
    return pose_2d


def move_to_center(pose_2d):
    '''
    Move normalized pose to the center (0.5, 0.5).

    :pose_2d: pose kpts with hom coordinate
    '''
    # TODO: Use move_2d_to_origin.
    max_coord = np.amax(pose_2d, axis=0)
    min_coord = np.amin(pose_2d, axis=0)
    mid_point = (max_coord + min_coord) / 2.
    pose_2d -= mid_point
    pose_2d += np.array([0.5, 0.5, 0.])
    return pose_2d


def move_from_center(pose_2d, y_axis=False):
    '''
    Move normalized pose from the center.
    '''
    trans_x = np.random.uniform(-0.2, 0.2, 1)
    pose_2d[:, 0] += trans_x
    return pose_2d


def create_look_at_matrix(x, y, z):
    from_ = np.array([x, y, z], dtype=np.float32)
    to = np.array([0., 0., 0.], dtype=np.float32)
    tmp = np.array([0., 1., 0.], dtype=np.float32)
    forward = (from_ - to)
    forward = forward / np.linalg.norm(forward)
    right = np.cross(tmp, forward)
    right = right / np.linalg.norm(right)
    up = np.cross(forward, right)

    R = [
        [right[0], up[0], forward[0]],
        [right[1], up[1], forward[1]],
        [right[2], up[2], forward[2]],
        [0.  ,     y   , -RADIUS   ]
    ]
    return np.transpose(R)


def sample_projection_matrix(orient_x=False, orient_z=False):
    # NOTE: Z-axis is depth, Y-axis is height.
    # TODO: Implement random Z-axis.
    NUM_VIEWS = 500
    step = 2 * np.pi / NUM_VIEWS
        
    angle = step * random.randint(0, NUM_VIEWS)
    if angle < np.pi / 2:
        z = -RADIUS / np.sqrt(np.tan(angle) ** 2 + 1)
        x = z * np.tan(angle)
    elif angle < np.pi:
        angle -= np.pi / 2
        x = -RADIUS / np.sqrt(np.tan(angle) ** 2 + 1)
        z = -x * np.tan(angle)
    elif angle < 3 * np.pi / 2:
        angle -= np.pi
        z = RADIUS / np.sqrt(np.tan(angle) ** 2 + 1)
        x = z * np.tan(angle)
    else:
        angle -= 3 * np.pi / 2
        x = RADIUS / np.sqrt(np.tan(angle) ** 2 + 1)
        z = -x * np.tan(angle)
            
    RT = create_look_at_matrix(x, 0., z)
    P = np.dot(np.array(K), RT)
    return P


def project(kpts_3d, proj_mat):
    ones_vector = np.ones((kpts_3d.shape[0], 1), dtype=np.float32)
    kpts_3d = np.hstack((kpts_3d, ones_vector))
    kpts_2d = np.dot(kpts_3d, proj_mat.transpose())
    last_row = kpts_2d[:, 2].reshape((kpts_2d.shape[0]), 1)
    kpts_2d_hom = np.multiply(kpts_2d, 1. / last_row)
    #print(kpts_2d_hom)
    return kpts_2d_hom.transpose(0, 1)


if __name__ == '__main__':
    #start_time = time()
    #P = generate_random_projection()
    Ps = generate_uniform_projections(5)
    
    #print('Generating random projection: {}'.format(time() - start_time))
    #P = create_projection_matrix(-90, 0)
    with open('dataset/3dpeople/train/woman17/02_04_jump/camera01/0026.txt') as kpt_f:
        lines = [x[:-1] for x in kpt_f.readlines()]
        kpts = [[float(x) for x in y.split(' ')] for y in lines[1:]]
        #kpts_3d = np.array([x[3:] for x in kpts]).swapaxes(0, 1)
        kpts_3d = torch.tensor([x[3:] for x in kpts]).transpose(0, 1)
    #start_time = time()
    for idx in range(5):
        kpts_2d = project(kpts_3d, Ps[idx])
        img = np.zeros((H, W, 3), np.uint8)
        img = draw_2d(kpts_2d, img)
        #img = draw_2d(kpts, img)
        cv2.imshow('2d keypoints', img)
        cv2.waitKey(0)
    #print('Projecting: {}'.format(time() - start_time))

