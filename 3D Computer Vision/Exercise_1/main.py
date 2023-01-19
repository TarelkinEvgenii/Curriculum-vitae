#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""Authors:
    Tarelkin Evgenii
    Steba Oxana"""

# In[2]:


import os, sys
import numpy as np
import cv2 as cv
import scipy.io as io
import matplotlib.pyplot as plt
import copy

# In[3]:


# path to new folders with result images
path1 = "result_without_correction"
os.mkdir(path1)
path2 = "result_with_correction"
os.mkdir(path2)


# In[4]:


def project_points(X, K, R, T, distortion_flag=False, distortion_params=None):
    """Project points from 3d world coordinates to 2d image coordinates.
    Your code should work with considering distortion and without
    considering distortion parameters.
    """
    # distortion parameters
    k1 = distortion_params[0]
    k2 = distortion_params[1]
    k5 = distortion_params[4]

    # calculation of points without distortion
    # translation points coordinates between world coordinate system and camera coordinate system
    x = R.dot(X)
    xc = T[0, 0] + x[0]
    yc = T[1, 0] + x[1]
    zc = T[2, 0] + x[2]
    X_C = [xc, yc, zc]
    # calculation homogeneous coordinates
    u_1 = K.dot(X_C)
    # converting from homogeneous coordinates
    x0 = u_1[0] / u_1[2]
    y0 = u_1[1] / u_1[2]
    x_2d = [x0, y0]
    x_1 = [x0, y0, 1]
    if distortion_flag == True:
        # calculation of points with distortion
        # point normalization
        xn = np.linalg.inv(K).dot(x_1)
        # calculation of radial distortion
        r = xn[0] ** 2 + xn[1] ** 2
        r2 = r ** 2
        r3 = r ** 3
        distor = 1 + k1 * r + k2 * r2 + k5 * r3
        # applying of radial distortion
        xd = xn[0] * distor
        yd = xn[1] * distor
        XD = [xd, yd, 1]
        # point denormalization
        Xdn = K.dot(XD)
        xdn = Xdn[0] / Xdn[2]
        ydn = Xdn[1] / Xdn[2]
        x_2d = [xdn, ydn]
    return x_2d


# In[5]:


def project_and_draw(img, X_3d, K, R, T, distortion_flag,
                     distortion_parameters):
    """
    call "project_points" function to project 3D points to camera coordinates
    draw the projected points on the image and save your output image here
    # save your results in a separate folder named "results"
    # Your implementation goes here!

    """
    # cycle to process all points in images
    for i in range(0, 25):
        for f in range(0, 2):
            image = copy.copy(img[i])
            for j in range(0, 40):
                if f == 1:
                    distortion_flag = True
                else:
                    distortion_flag = False
                points_2D = []
                points_2D = project_points(X_3d[i, j], K, R[i], T[i],
                                           distortion_flag,
                                           distortion_parameters)
                # center of circle to draw
                center = (int(points_2D[0]), int(points_2D[1]))
                # radius of circle
                radius = 4
                # line thickness
                thickness = 1
                # drawing circles on images
                if distortion_flag == True:
                    color = (0, 255, 0)
                    image = cv.circle(image, center, radius, color, thickness)
                    path = "./" + path2
                    img_name = "img" + str(i) + "_2.jpg"
                    cv.imwrite(os.path.join(path, img_name), image)
                    cv.waitKey(0)
                else:
                    color = (0, 0, 255)
                    image = cv.circle(image, center, radius, color, thickness)
                    path = "./" + path1
                    img_name = "img" + str(i) + "_1.jpg"
                    cv.imwrite(os.path.join(path, img_name), image)
                    cv.waitKey(0)
        cv.destroyAllWindows()
    print(
        "Your calculation is done. Check the results in the folders 'result_without_correction' and 'result_with_correction', please.")


# In[6]:


if __name__ == '__main__':
    base_folder = './data/'

    # Consider distorition
    dist_flag = True

    # Load the data
    # There are 25 views/or images/ and 40 3D points per view
    data = io.loadmat('./data/ex_1_data.mat')

    # 3D points in the world coordinate system
    X_3D = data['x_3d_w']  # shape=[25, 40, 3]

    # Translation vector: as the world origin is seen from the camera coordinates
    TVecs = data['translation_vecs']  # shape=[25, 3, 1]

    # Rotation matrices: project from world to camera coordinate frame
    RMats = data['rot_mats']  # shape=[25, 3, 3]

    # five distortion parameters
    dist_params = data['distortion_params']

    # K matrix of the cameras
    Kintr = data['k_mat']  # shape 3,3

    imgs_list = [cv.imread(base_folder + str(i).zfill(5) + '.jpg') for i in
                 range(TVecs.shape[0])]
    imgs = np.asarray(imgs_list)
    project_and_draw(imgs, X_3D, Kintr, RMats, TVecs, dist_flag, dist_params)

# In[ ]:


# In[ ]:
