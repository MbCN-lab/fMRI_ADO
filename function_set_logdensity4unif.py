############################################
# Adaptive Design Optimization             #
# Function set: Log Density Functions V4   #
# Giwon Bahg                               #
# June 26, 2017                            #
############################################

# This file defines the prior and likelihood functions
# assuming uniform priors for all joint model parameters.

import numpy as np
import math
import scipy.stats as st
from scipy.special import erf

# To implement cumulative normal distribution:
def _norm_cdf(x, mu, sigma):
    return 0.5 * (1 + erf((x-mu)/(sigma*np.sqrt(2))))


def logdensprior(b, Rmax, c50, delta, prior_range):
    prior_mean = np.array([0, 1, 0.5, 0.5])
    prior_sd = np.array([100, 100, 100, 100])

    lprior = np.empty_like(b, dtype = np.longdouble)

    # Constraint
    crf_c50 = b + (Rmax * c50 ** 2) / (2 * c50 ** 2)
    bad_ind = (b >= crf_c50) | (crf_c50 >= Rmax) | (delta <= 0)
    bad_ind = bad_ind | (b < prior_range[0]) | (b > prior_range[1])
    bad_ind = bad_ind | (Rmax < prior_range[2]) | (Rmax > prior_range[3])
    bad_ind = bad_ind | (c50 < 0.) | (c50 > 1.)
    bad_ind = bad_ind | (delta < prior_range[4]) | (delta > prior_range[5])
    good_ind = ~bad_ind
    num_good = good_ind.sum()

    lprior[bad_ind] = -np.inf
    # st.truncnorm.logpdf(input, min, max, mean, sd)
    lprior[good_ind] = (st.truncnorm.logpdf(b[good_ind],
                                            a=(prior_range[0] - prior_mean[0]) / prior_sd[0],
                                            b=(prior_range[1] - prior_mean[0]) / prior_sd[0],
                                            loc=prior_mean[0], scale=prior_sd[0]) +
                        st.truncnorm.logpdf(Rmax[good_ind],
                                            a=(prior_range[2] - prior_mean[1]) / prior_sd[1],
                                            b=(prior_range[3] - prior_mean[1]) / prior_sd[1],
                                            loc=prior_mean[1], scale=prior_sd[1]) +
                        st.truncnorm.logpdf(c50[good_ind],
                                            a=(-prior_mean[2]) / prior_sd[2],
                                            b=(1 - prior_mean[2]) / prior_sd[2],
                                            loc=prior_mean[2], scale=prior_sd[2]) +
                        st.truncnorm.logpdf(delta[good_ind],
                                            a=(prior_range[4] - prior_mean[3]) / prior_sd[3],
                                            b=(prior_range[5] - prior_mean[3]) / prior_sd[3],
                                            loc=prior_mean[3], scale=prior_sd[3]))
    return lprior


def logdenspost_prop(contrast_space, b, Rmax, c50, delta, designgrid, beta1, beta2, behav, prior_range):

    # allocate for results
    llike = np.empty_like(b, dtype = np.longdouble)

    # Constraint
    crf_c50 = b + (Rmax * c50 ** 2) / (2 * c50 ** 2)
    bad_ind = (b >= crf_c50) | (crf_c50 >= Rmax) | (delta <= 0)
    bad_ind = bad_ind | (b < prior_range[0]) | (b > prior_range[1])
    bad_ind = bad_ind | (Rmax < prior_range[2]) | (Rmax > prior_range[3])
    bad_ind = bad_ind | (c50 < 0.) | (c50 > 1.)
    bad_ind = bad_ind | (delta < prior_range[4]) | (delta > prior_range[5])
    good_ind = ~bad_ind
    num_good = good_ind.sum()

    # set the log likes for the bad
    llike[bad_ind] = -np.inf
    if num_good == 0:
        return llike

    # Likelihood of the joint model:
    contrast_prop = contrast_space[designgrid[good_ind]]
    obs_beta_prop = np.concatenate([beta1[good_ind, np.newaxis], beta2[good_ind, np.newaxis]], axis=1)
    obs_behav_prop = behav[good_ind, np.newaxis]

    ## Neural likelihood: Naka-Rushton equation
    crf_pred = (b[good_ind, np.newaxis] + (Rmax[good_ind, np.newaxis] * contrast_prop ** 2) /
                (c50[good_ind, np.newaxis] ** 2 + contrast_prop ** 2))
    loglike_neural = st.norm.logpdf(obs_beta_prop, loc = crf_pred, scale = delta[good_ind, np.newaxis] / np.sqrt(2)).sum(axis = 1)

    ## Behavioral likelihood: Thurstonian decision model
    diff_beta_prop = np.diff(crf_pred.reshape(crf_pred.shape[0],
                                              crf_pred.shape[1] / 2,
                                              2), axis=2)[:, :, 0]
    prob_resp_1 = 1.0 - _norm_cdf(0, diff_beta_prop, delta[good_ind, np.newaxis])

    loglike_behav = st.binom.logpmf(obs_behav_prop, n = 1, p = prob_resp_1).sum(axis=1)

    temp_like = loglike_behav + loglike_neural

    # # The following lines were included in logdenspost_data to avoid "divide by zero" error
    # # and added here as well for consistency in density functions.
    # minval = (temp_like[np.where(np.isfinite(temp_like))[0]]).min()
    # temp_like[np.where(np.isneginf(temp_like))[0]] = minval - 1e-300
    # temp_like[np.where(np.isnan(temp_like))[0]] = minval - 1e-300

    llike[good_ind] = temp_like

    return llike

def logdenspost_data(given_contrast_temp, obs_beta_temp, obs_behav_temp, now_where,
                     b, Rmax, c50, delta, prior_range):

    given_contrast = np.array(given_contrast_temp)
    obs_beta = np.array(obs_beta_temp)
    obs_behav = np.array(obs_behav_temp)

    # allocate for results
    llike = np.empty_like(b, dtype = np.longdouble)

    # Constraint
    crf_c50 = b + (Rmax * c50 ** 2) / (2 * c50 ** 2)
    bad_ind = (b >= crf_c50) | (crf_c50 >= Rmax) | (delta <= 0)
    bad_ind = bad_ind | (b < prior_range[0]) | (b > prior_range[1])
    bad_ind = bad_ind | (Rmax < prior_range[2]) | (Rmax > prior_range[3])
    bad_ind = bad_ind | (c50 < 0.) | (c50 > 1.)
    bad_ind = bad_ind | (delta < prior_range[4]) | (delta > prior_range[5])
    good_ind = ~bad_ind
    num_good = good_ind.sum()

    # set the log likes for the bad
    llike[bad_ind] = -np.inf
    if num_good == 0:
        return llike

    # prior
    log_prior = np.repeat((np.log(1/(prior_range[1] - prior_range[0])) + # b
                           np.log(1/(prior_range[3] - prior_range[2])) + # Rmax
                           np.log(1/(1 - 0)) +                           # c50
                           np.log(1/(prior_range[5] - prior_range[4]))), # delta
                          b[good_ind].size)

    if now_where == 0:
        llike[good_ind] = log_prior
        return llike
    else:
        # Likelihood
        # given from global: contrast, obs_beta, obs_behav
        given_contrast_arr = given_contrast[np.newaxis, :]*np.ones((num_good, 1))
        obs_beta_arr = obs_beta[np.newaxis, :]*np.ones((num_good, 1))
        if (len(obs_behav.shape) == 0):
            obs_behav_arr = obs_behav[np.newaxis]*np.ones((num_good, 1))
        else:
            obs_behav_arr = obs_behav[np.newaxis, :]*np.ones((num_good, 1))


        # Neural likelihood
        crf_pred = (b[good_ind, np.newaxis] + (Rmax[good_ind, np.newaxis] * given_contrast_arr ** 2) /
                    (c50[good_ind, np.newaxis] ** 2 + given_contrast_arr ** 2))
        loglike_neural = st.norm.logpdf(obs_beta_arr, loc=crf_pred,
                                        scale=delta[good_ind, np.newaxis] / np.sqrt(2)).sum(axis=1)

        # # Behavioral likelihood
        diff_beta_arr = np.diff(crf_pred.reshape(crf_pred.shape[0],
                                                 crf_pred.shape[1] / 2,
                                                 2), axis=2)[:, :, 0]


        prob_resp_1 = 1.0 - _norm_cdf(0, mu = diff_beta_arr, sigma = delta[good_ind, np.newaxis])
        loglike_behav = st.binom.logpmf(obs_behav_arr, n=1, p=prob_resp_1).sum(axis=1)

        temp_like = loglike_neural + loglike_behav

        # # To avoid "divide by zero" error:
        # minval = (temp_like[np.where(np.isfinite(temp_like))[0]]).min()
        # temp_like[np.where(np.isneginf(temp_like))[0]] = minval - 1e-300
        # temp_like[np.where(np.isnan(temp_like))[0]] = minval - 1e-300

        llike[good_ind] = temp_like + log_prior

    return llike