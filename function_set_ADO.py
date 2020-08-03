############################################
# Adaptive Design Optimization             #
# Function set: ADO Wrapper                #
# Giwon Bahg                               #
# January 24, 2018                         #
############################################

# This file contains a wrapper function of ADO that is called in the experiment program directly.
# Evaluation of the posterior density relies on "function_set_logensity4unif.py"

import numpy as np
from function_set_logdensity4unif import logdenspost_prop, logdenspost_data
import time

### ADO functions

def finite_min(array):
    return (array[np.where(np.isfinite(array))[0]]).min()

def replace_neginf(array):
    temp = array
    minval = (array[np.where(np.isfinite(array))[0]]).min()
    temp[np.where(np.isneginf(temp))[0]] = minval - 1e-300
    return temp

def normalize(array):
    temp = array
    minval = (array[np.where(np.isfinite(temp))[0]]).min()
    temp[np.where(np.isneginf(temp))[0]] = minval - 1e-300
    temp = np.exp(temp)
    temp = temp / temp.sum()
    return temp

def normalize_for_d_and_y(array, propgrid, prop_idx):
    out = np.zeros_like(array, dtype = np.longdouble)
    for i in prop_idx:
        temp_idx = np.where(propgrid == prop_idx[i])
        temp = array[temp_idx]
        minval = (array[np.where(np.isfinite(temp))[0]]).min()
        temp[np.where(np.isneginf(temp))[0]] = minval - 1e-300
        temp = np.exp(temp)
        out[temp_idx] = temp / temp.sum()
    return out

def joint_density(array, designgrid, designidx):
    temp = np.zeros_like(array, dtype = np.longdouble)
    for i in designidx:
        targetidx = np.where(designgrid == i)[0]
        temp[targetidx] = normalize(array[targetidx])
    return temp

def aggregate_gu(gu, designgrid, designidx, cgrid, dgrid, option):
    temp = np.zeros(designidx.size, dtype = np.longdouble)
    out_of_range = (cgrid < 0) | (cgrid > 1) | (dgrid <= 0)
    if (option == 'sum'):
        for i in designidx:
            targetidx = np.where((designgrid == i) & (~out_of_range))[0]
            temp[i] = np.nansum(gu[targetidx])
    elif (option == 'mean'):
        for i in designidx:
            targetidx = np.where((designgrid == i) & (~out_of_range))[0]
            temp[i] = np.nanmean(gu[targetidx])
    return temp

def ADO_wrapper_new(contrast, obs_beta, obs_behav, now_where, design_space, design_idx,
                    prmt_b, prmt_Rmax, prmt_c50, prmt_delta,
                bgrid, Rgrid, cgrid, dgrid, designgrid, b1grid, b2grid, bhvgrid, propgrid, prop_idx, prior_range):

    # invalid_trial = list()
    # for i in range(len(obs_behav)):
    #     if (obs_behav[i] == ''):
    #         invalid_trial.append(i)
    # for i in range(len(obs_behav)):
    #     if (obs_)

    start_time = time.time()
    post_prop_only = logdenspost_prop(design_space, bgrid, Rgrid, cgrid, dgrid, designgrid, b1grid, b2grid, bhvgrid, prior_range)

    post_data = logdenspost_data(contrast, obs_beta, obs_behav, now_where, prmt_b, prmt_Rmax, prmt_c50, prmt_delta, prior_range)
    post_data = np.tile(post_data, bgrid.size / post_data.size)

    post_prop = post_prop_only + post_data

    # Before normalizing: Replace all -Inf with a finite infinitesimal value
    minval_data = (post_data[np.where(np.isfinite(post_data))[0]]).min()
    post_data[np.where(np.isneginf(post_data))[0]] = minval_data - 1e-300

    minval_prop = (post_prop[np.where(np.isfinite(post_prop))[0]]).min()
    post_prop[np.where(np.isneginf(post_prop))[0]] = minval_prop - 1e-300

    # P(theta | d, y)
    # post_prop: un-normalized posterior density in a log scale
    # normalize for each "proposal design"
    temp_post_prop_array = post_prop.reshape(prop_idx.size, propgrid.size / prop_idx.size)
    norm_post_prop_array = np.apply_along_axis(normalize, 1, temp_post_prop_array)
    norm_post_prop = norm_post_prop_array.ravel()

    # P(theta, y | d)
    joint_given_d = joint_density(post_prop, designgrid, design_idx)

    # Local utility: log( P(theta | d, y) / P(theta) )
    lu = np.log(norm_post_prop) - post_data

    # Global utility: lu * joint_given_d
    gu = lu * joint_given_d
    gu_aggregated = aggregate_gu(gu, designgrid, design_idx, cgrid, dgrid, 'mean')
    print(gu_aggregated)

    next_pool = np.argwhere(gu_aggregated == np.amax(gu_aggregated)).ravel()
    next_design = np.random.choice(next_pool, size=1)[0]

    ADO_time = time.time() - start_time
    print("Computation time: ", ADO_time)
    return next_design.tolist()#, norm_post_data.tolist()
