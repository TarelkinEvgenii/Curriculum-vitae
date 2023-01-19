#!/usr/bin/env python
# coding: utf-8

# In[101]:


"""Authors:
    Tarelkin Evgenii
    Steba Oxana"""

# In[102]:


import os, sys
import numpy as np
import scipy.io as io
import math
import re
from numpy.linalg import svd


# In[103]:


def compute_relative_rotation(H, K):
    # funcion of calculation Rrel

    # calcultion of inverted matrix of intrinsic parameters
    K_inv = np.linalg.inv(K)

    # calculation of lambda and rotation matrix
    lR = K_inv.dot(H.dot(K))

    # finding lengths of r1 and r2
    r1_len = math.sqrt(lR[0, 0] ** 2 + lR[0, 1] ** 2 + lR[0, 2] ** 2)
    r2_len = math.sqrt(lR[1, 0] ** 2 + lR[1, 1] ** 2 + lR[1, 2] ** 2)

    # calculation of lambda using r1 and r2
    l1 = r1_len / 1
    l2 = r2_len / 1
    # calculation of average lambda value
    l = (l1 + l2) / 2

    # calcultion of relative rotation matrix
    Rrel = lR * (1 / l)
    print("Relative rotation Rrel: \n", Rrel)

    # check rotation properties
    # det(Rel)=1
    detR = round(np.linalg.det(Rrel), 2)
    # R^-1=R^T
    R_inv = np.linalg.inv(Rrel)
    R_T = Rrel.transpose()
    for i in range(0, 3):
        for j in range(0, 3):
            R_inv[i, j] = round(R_inv[i, j], 5)
            R_T[i, j] = round(R_T[i, j], 5)
        # comparison of matrices
    eq = np.array_equal(R_inv, R_T)
    # if (wrong)
    if detR != 1 and eq == False:
        # correction
        U, W, VT = svd(Rrel)

        # calculation of new rotation matrix
        Rnew = U.dot(VT)
        print(
            "\nThe rotation matrix has changed. New relative rotation Rrel: \n",
            Rnew)


# In[104]:


def compute_pose(H, K):
    # calculation of invered matrix
    K_inv = np.linalg.inv(K)
    # calculation of lambda, rotation matrix and translation vector
    lRt = K_inv.dot(H)

    # finding r1 and r2
    r1 = [lRt[0, 0], lRt[0, 1], lRt[0, 2]]
    r2 = [lRt[1, 0], lRt[1, 1], lRt[1, 2]]

    # calculation of r1 and r2 lengths
    r1_len = math.sqrt(r1[0] ** 2 + r1[1] ** 2 + r1[2] ** 2)
    r2_len = math.sqrt(r2[0] ** 2 + r2[1] ** 2 + r2[2] ** 2)

    # calculation of lambda from r1 and r2
    l1 = r1_len / 1
    l2 = r2_len / 1
    # calculation of average lambda
    l = (l1 + l2) / 2

    # calculation of r3 using cross-product of r1 and r2
    r3 = np.cross(r1, r2)

    # calculation of translation vector
    t = (1 / l) * lRt[2]

    # calculation of rotation matrix
    R = np.array([r1,
                  r2,
                  [r3[0], r3[1], r3[2]]])

    print("Rotation matrix R: \n", R)
    print("\nTranslation vector t: \n", t)


# In[105]:


if __name__ == '__main__':
    base_folder = './data/'

    # Load the data
    data = io.loadmat('./data/ex2.mat')

    # ax from the K matrix
    ax = data['alpha_x']

    # skew parameter
    s = data['s']

    # x0 from K matrix
    x0 = data['x_0']

    # ay from the K matrix
    ay = data['alpha_y']

    # y0 rom K matrix
    y0 = data['y_0']

    # K matrix of the cameras
    Kintr = np.array([[ax[0, 0], s[0, 0], x0[0, 0]],
                      [0, ay[0, 0], y0[0, 0]],
                      [0, 0, 1]])

    # H homograpies
    H1 = data['H1']
    H2 = data['H2']
    H3 = data['H3']
    print("\nH1:\n")
    compute_relative_rotation(H1, Kintr)
    print("\nH2:\n")
    compute_relative_rotation(H2, Kintr)
    print("\nCalculation of R and t from H3:\n")
    compute_pose(H3, Kintr)

# In[ ]:
