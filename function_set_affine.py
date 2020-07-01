############################################
# Adaptive Design Optimization             #
# Function set: Affine transformation      #
# Giwon Bahg                               #
# June 13, 2017                            #
############################################

# A function set for eigendecomposition-based dynamic gridding.

from __future__ import print_function
import numpy as np

def affine_trans_grid(theta, pct):
    post_dens = np.zeros([theta.shape[0] * theta.shape[2], theta.shape[1]])
    for i in range(theta.shape[1]):
        post_dens[:,i] = theta[:,i,:].ravel()

    prmt_means = post_dens.mean(axis = 0)
    prmt_cov = np.cov(post_dens.transpose())
    eigval, eigvec = np.linalg.eig(prmt_cov)

    rotateMat = np.matmul(eigvec, np.diag(np.sqrt(eigval)))
    invmapMat = np.linalg.inv(rotateMat)

    orth = np.matmul(post_dens, invmapMat.transpose())
    temp_axes = np.percentile(orth, pct, axis = 0).transpose()

    temp_axis3, temp_axis4, temp_axis2, temp_axis1 = np.meshgrid(temp_axes[2], temp_axes[3], temp_axes[1], temp_axes[0])
    temp_axis1 = temp_axis1.ravel()
    temp_axis2 = temp_axis2.ravel()
    temp_axis3 = temp_axis3.ravel()
    temp_axis4 = temp_axis4.ravel()
    temp_axis_mat = np.column_stack([temp_axis1, temp_axis2, temp_axis3, temp_axis4])

    new_axis_grid = np.matmul(temp_axis_mat, rotateMat.transpose())

    # bad_c50_min = (new_axis_grid[:,2] < 0)
    # bad_c50_max = (new_axis_grid[:,2] > 1)
    # bad_delta = (new_axis_grid[:, 3] <= 0)
    #
    # new_axis_grid[bad_c50_min] = np.min(new_axis_grid[~bad_c50_min, 2])
    # new_axis_grid[bad_c50_max] = np.max(new_axis_grid[~bad_c50_max, 2])
    # new_axis_grid[bad_delta,3] = np.min(new_axis_grid[~bad_delta, 3])

    return new_axis_grid[:,0], new_axis_grid[:,1], new_axis_grid[:,2], new_axis_grid[:,3]