# -*- coding: utf-8 -*-
"""
Authors:
    Tarelkin Evgenii
    Steba Oxana
"""

import os, sys
import scipy.io as io
import numpy as np
from matplotlib import pyplot as plt
import cv2
import random
import copy
from numpy.random import rand


def algebr_dist(x, l):
    a, b, c = l.reshape(3).tolist()
    x, y = x.reshape(2).tolist()
    return np.abs(a * x + b * y + c) / np.sqrt(a * a + b * b)


def feature_match(K0, K1, R1, T1, corn0, corn1, im1, basefolder):
    # fundamental matrix
    K1_iT = (np.linalg.inv(K1)).transpose()
    K0_inv = np.linalg.inv(K0)
    tx = np.array([[0, -T1[0, 2], T1[0, 1]],  # shape=[3, 3]
                   [T1[0, 2], 0, -T1[0, 0]],
                   [-T1[0, 1], T1[0, 0], 0]])
    F = ((K1_iT.dot(tx)).dot(R1)).dot(K0_inv)

    # epipolar lines in im01
    wid = 4752
    for i in range(0, len(corn0) - 1):
        point = [corn0[i, 0], corn0[i, 1], 1]
        l = F.dot(point)  # corn0[0]
        x0 = 0
        y0 = int(-l[2] / l[1])
        x1 = wid
        y1 = int(-(l[2] + l[0] * wid) / l[1])
        img1 = cv2.line(im1, (x0, y0), (x1, y1), (0, 0, 255), 7)

    print("Fundamental matrix F:\n", F)

    cv2.imwrite('./results/epilines.jpg', img1)

    plt.imshow(img1)
    return F


def rans(im0, im1, corn0, corn1, F):
    result = cv2.vconcat([im0, im1])

    for i in range(0, len(corn0) - 1):
        point0 = np.array([corn0[i, 0], corn0[i, 1], 1])
        l = F.dot(point0)
        color = (255, 0, 0)
        result = cv2.circle(result, (int(corn0[i, 0]), int(corn0[i, 1])), 12,
                            color, 9)
        for j in range(0, len(corn1) - 1):
            point1 = np.array([corn1[j, 0], corn1[j, 1], 1])
            point1 = point1[:, np.newaxis].T
            col = tuple(np.random.randint(0, 255, 3).tolist())
            x_min = corn1[0]
            for x in corn1:
                if algebr_dist(x, l) < algebr_dist(x_min, l):
                    x_min = x
            cv2.line(result, (int(Corners0[i][0]), int(Corners0[i][1])),
                     (int(x_min[0]), int(3168 + x_min[1])), col, 10)
    cv2.imwrite("./results/matches.jpg", result)


if __name__ == '__main__':
    base_folder = './data/'

    # Load the data
    data = io.loadmat('./data/data.mat')
    # read data in COLAB
    # import scipy.io
    # data = scipy.io.loadmat('data.mat')

    # K matrices of the 2 cameras
    Kintr0 = data['K_0']  # shape=[3, 3]
    Kintr1 = data['K_1']  # shape=[3, 3]

    # Rotation matrices
    RMat0 = np.array([[1, 0, 0],  # shape=[3, 3]
                      [0, 1, 0],
                      [0, 0, 1]])
    RMat1 = data['R_1']  # shape=[3, 3]

    # Translation vectors
    TVec0 = np.array([[0, 0, 0]])  # shape=[3, 1]
    TVec1 = data['t_1']  # shape=[3, 1]

    # 3D points in the image
    Corners0 = data['cornersCam0']  # shape=[80, 2]
    Corners1 = data['cornersCam1']  # shape=[80, 2]

    img0 = cv2.imread('./data/Camera00.jpg')
    img1 = cv2.imread('./data/Camera01.jpg')
    im1 = copy.copy(img1)

    img00 = copy.copy(img0)
    img11 = copy.copy(img1)

    F = feature_match(Kintr0, Kintr1, RMat1, TVec1, Corners0, Corners1, im1,
                      base_folder)
    rans(img00, img11, Corners0, Corners1, F)

"""use F from 1st
each ideal point - mke epipolar line
check the equztion
x`T*l`=0
"""
