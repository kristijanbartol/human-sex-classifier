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


DATASET_DIR = './dataset/'


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


def show_top(dataset, img_dims, grid_dims, show, show_image, num_top=100):
    dataset_dir = os.path.join(DATASET_DIR, dataset)
    best_json = os.path.join(dataset_dir, f'report_best_{num_top}.json')
    worst_json = os.path.join(dataset_dir, f'report_worst_{num_top}.json')
    num_samples = int((grid_dims[0] * grid_dims[1]) / 2)

    best_idxs = [x[0] for x in best_json.items()[:num_samples]]
    worst_idxs = [x[0] for x in worst_json.items()[:num_samples]]
    idxs = np.array(best_idxs + worst_idxs, dtype=np.uint32)
    
    tiles = np.zeros((img_dims[0] * grid_dims[0],
        img_dims[1] * grid_dims[1]), dtype=np.uint8)
    if show_image:
        tiles = load_images(tiles, best_idxs, worst_idxs, grid_dims)
    tiles = load_kpts(tiles, best_idxs, worst_idxs, grid_dims)

    if show:
        cv2.imshow('Best/worst result tiles', tiles)
        cv2.waitKey(0)
    cv2.imsave(os.path.join(dataset_dir, 'top.png'), tiles)


def show_subsets(dataset, mode, grid, show, show_image):
    dataset_dir = os.path.join(DATASET_DIR, dataset)
    if mode == 'subset':
        json_path = os.path.join(dataset_dir, 'test_idxs.json')
    else:
        json_path = os.path.join(dataset_dir, f'test_{mode}_idxs.json')
    num_samples = int((grid_dims[0] * grid_dims[1]) / 2)

    with open(json_path) as fjson:
        json_data = json.load(fjson)
    # TODO: For every subset, show samples.
    for key in json_data:
        pass


def init_parser():
    parser = argparse.ArgumentParser(
            description='Visualize stacked poses and original images.')
    parser.add_argument('--dataset', type=str,
            choices=['3dpeople', 'peta']
            help='which dataset (directory) to visualize')
    parser.add_argument('--mode', type=str,
            choices=['subset', 'action', 'subject', 'top']
            help='which reports to visualize')
    parser.add_argument('--grid_dims', type=int, nargs='+',
            help='maximal dimensions of the pose grid (X x Y)')
    parser.add_argument('--img_dims', type=int, nargs='+',
            help='image dimensions (tiles in the grid)')
    parser.add_argument('--show', dest='show', action='store_true',
            help='show (display) the result on the screen')
    parser.add_argument('--show_image', dest='show_image', action='store_true',
            help='use original images as backgrounds')

    args = parser.parse_args()
    return args


def check_args(args):
    if args.dataset == 'peta' and (args.mode == 'action' or \
            args.mode == 'subject'):
        raise Exception('Use SUBSET or TOP mode with PETA dataset.')
    if args.dataset == '3dpeople' and args.mode == 'subset':
        raise Exception('Use ACTION, SUBJECT or TOP mode with 3DPeople.')
    return args


if __name__ == '__main__':
    args = check_args(init_parser())
    if args.mode == 'top':
        show_top(args.dataset, args.img_size, args.grid, args.show, 
                args.show_image)
    else:
        show_subsets(args.dataset, args.img_size, args.mode, args.grid, 
                args.show, args.show_image)

