from typing import List
from bisect import bisect_left

import scipy
import numpy as np

from febrisk.math import is_psd
from febrisk.dist_fit import DistFitter
from febrisk.math import PCA


def chol_psd(sigma):
    """
    Apply Cholesky Factorization to matrix sigma.
    Sigma is expected to be PD or PSD.
    """
    root = np.full(sigma.shape, 0.0, dtype='float64')

    # loop over columns
    ncols = root.shape[1]
    for j in range(ncols):

        # 1. the diagonal value
        # Ljj = sqrt(Ajj - (Lj1^2 + ... + Ljj-1^2))
        diag_val = sigma[j, j] - root[j, :j] @ root[j, :j].T
        if -1e-8 <= diag_val <= 0:
            diag_val = 0.0
        elif diag_val < -1e-8:
            raise ValueError("The matrix is not positive semi-definite!")
        root[j, j] = np.sqrt(diag_val)

        # 2. non-diagonal values
        # If the diagnal value is 0, leave the rest on the column to be 0
        if root[j, j] == 0:
            continue
        for i in range(j+1, ncols):
            root[i, j] = (sigma[i, j] - root[i, :j] @ root[j, :j].T) / root[j, j]

    return root


class CholeskySimulator:

    def __init__(self, covariance):
        self.root = chol_psd(covariance)

    def simulate(self, nsample):
        """"
        Returns a simulated dataset that follows the normal distribution
        with shape (# of data, # of dims)
        """
        return (self.root @ np.random.randn(self.root.shape[1], nsample)).T


class PCASimulator:

    def __init__(self, sigma):
        self.pca = PCA(sigma)

    def factorize(self, explained, verbose):
        explained = min(explained, 1)
        explained = max(explained, 0)

        # find the index of the minimum cumulative_evr
        # that is greater than or equals to explained
        idx = bisect_left(self.pca.cumulative_evr, explained)
        eig_vals = self.pca.explained_variance[:idx+1]
        eig_vecs = self.pca.eig_vecs[:, :idx+1]

        if verbose:
            print(f"{self.pca.cumulative_evr[idx]*100:.2f}% total variance explained.\n" +
                  f"{idx+1} eigen value(s) are used.")

        return eig_vecs @ np.diag(np.sqrt(eig_vals))

    def simulate(self, nsample, explained=1, verbose=False):
        """
        Returns a simulated dataset that follows the normal distribution
        with shape (# of data, # of dims).
        """
        principal_components = self.factorize(explained, verbose)
        std_normal_random = np.random.randn(principal_components.shape[1], nsample)
        return (principal_components @ std_normal_random).T


class CopulaSimulator:

    def __init__(self, data, dists):
        assert data.shape[1] == len(dists)
        self.dists = dists
        
        # calculate quantiles' spearmanr
        quantiles = np.empty(data.shape)
        for i in range(data.shape[1]):
            quantiles[:, i] = self.dists[i].cdf(data[:, i])
        sp_corr = scipy.stats.spearmanr(quantiles, axis=0)[0]
        assert sp_corr.shape[1] == quantiles.shape[1], \
            "The size of correlation matrix doesn't match the number of variables"
        if not is_psd(sp_corr):
            raise ValueError("Spearman correlation matrix is not PSD!")
        
        self.spearmanr = sp_corr
        
    def simulate(self, nsample):
        simulator = CholeskySimulator(self.spearmanr)
        std_norm_vals = simulator.simulate(nsample)
        std_norm_quantiles = scipy.stats.norm(loc=0, scale=1).cdf(std_norm_vals)

        # for each column in standard normal cdfs, reverse them
        # to the actual value using correspondent distributions
        sim_vals = np.empty(shape=std_norm_quantiles.shape, dtype=float)
        for i in range(sim_vals.shape[1]):
            sim_vals[:, i] = self.dists[i].ppf(std_norm_quantiles[:, i])
        return sim_vals
