import os
import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
from copy import deepcopy
from scipy.spatial.transform import Rotation as rot
import torch
import h5py
import json
from random import random

from const import PELVIS, H36M_KPTS_15, H36M_PARTS_15, KPTS_17, BODY_PARTS_17, \
        KPTS_15, OPENPOSE_PARTS_15, RADIUS, K
#from data_utils import generate_random_projection, \
#        generate_uniform_projection_matrices, project, normalize_3d_numpy, \
#        generate_uniform_projections_torch


PEOPLE3D_H = 480
PEOPLE3D_W = 640


def draw_keypoints(kpts_2d, h, w):
    img = np.zeros((h, w, 3), np.uint8)

    for kpt_idx, kpt in enumerate(kpts_2d):
        kpt = tuple(kpt[:2])
        print(kpt)
        if kpt_idx == 5:
            img = cv2.circle(img, kpt, radius=1, color=(0, 255, 0), thickness=-1)
        else:
            img = cv2.circle(img, kpt, radius=1, color=(0, 0, 255), thickness=-1)

    cv2.imshow('2d keypoints', img)
    cv2.waitKey(0)


def draw_txt(txt_path):
    kpts_2d = []
    with open(txt_path) as f:
        lines = [x[:-1] for x in f.readlines()][1:]
    for line_idx, line in enumerate(lines):
        if line_idx + 1 in KPTS_15:
            kpts_2d.append([float(x) for x in line.split(' ')])
    kpts_2d = np.array([x[:3] for x in kpts_2d], dtype=np.float32)
    draw_keypoints(kpts_2d, 480, 640)


def draw_openpose(json_fpath, img_path):
    # TODO: Put original image as background.
    #img = np.zeros((1000, 1000, 3), dtype=np.uint8)
    img = cv2.imread(img_path)

    with open(json_fpath) as fjson:
        data = json.load(fjson)
    pose_2d = np.array(data['people'][0]['pose_keypoints_2d'], dtype=np.float32)
    pose_2d = np.delete(pose_2d, np.arange(2, pose_2d.size, 3))
    print(pose_2d)
    for idx in range(int(pose_2d.shape[0] / 2)):
        coord = (pose_2d[idx*2], pose_2d[idx*2+1])
        print(coord)
        img = cv2.circle(img, coord, radius=1, color=(0, 255, 0), thickness=-1)

    for part in OPENPOSE_PARTS_15:
        start_point = (pose_2d[part[0]*2], pose_2d[part[0]*2+1])
        end_point = (pose_2d[part[1]*2], pose_2d[part[1]*2+1])
        img = cv2.line(img, start_point, end_point, (255, 0, 0), thickness=1)

    cv2.imshow('2d keypoints', img)
    cv2.waitKey(0)


def init_parser():
    parser = argparse.ArgumentParser(
            description='Visualize stacked poses and original images.')
    parser.add_argument('--dataset', type=str,
            help='which dataset (directory) to visualize')
    parser.add_argument('--report', type=str,
            choices=['subset', 'action', 'subject', 'best', 'worst']
            help='which reports to visualize')
    parser.add_argument('--grid', type=int, nargs='+',
            help='maximal dimensions of the pose grid (X x Y)')
    parser.add_argument('--show_image', dest='show_image', action='store_true',
            help='use original images as backgrounds')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = init_parser()

