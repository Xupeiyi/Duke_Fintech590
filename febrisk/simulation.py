from typing import List
from bisect import bisect_left

import scipy
import numpy as np
import pandas as pd

from febrisk.psd import is_psd
from febrisk.dist_fit import DistFitter
from febrisk.statistics import PCA


def chol_psd(sigma):
    """
    Apply Cholesky Factorization to matrix sigma.
    Sigma is expected to be PD or PSD.
    """
    root = np.full(sigma.shape, 0.0, dtype='float64')

    # loop over columns
    n = root.shape[1]
    for j in range(n):

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
        for i in range(j+1, n):
            root[i, j] = (sigma[i, j] - root[i, :j] @ root[j, :j].T) / root[j, j]

    return np.matrix(root)


class CholeskySimulator:
    
    def __init__(self, covariance):
        self.root = chol_psd(covariance)
    
    def simulate(self, nsample):
        """"
        Returns a simulated dataset that follows the normal distribution
        with shape (# of dims, # of data)
        """
        return self.root @ scipy.random.randn(self.root.shape[1], nsample)


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
        with shape (# of dims, # of data).
        """
        L = self.factorize(explained, verbose)
        Z = scipy.random.randn(L.shape[1], nsample)
        return L @ Z


class CopulaSimulator:

    def __init__(self, dists=None, spearmanr=None):
        if dists and spearmanr is not None:
            assert len(dists) == spearmanr.shape[0]
        self.dists = dists if dists is not None else []
        self.spearmanr = spearmanr
    
    def _update_spearmanr(self, x):
        # calculate cdfs' spearmanr
        cdfs = np.empty(x.shape) 
        for i in range(x.shape[0]):
            cdfs[i, :] = self.dists[i].cdf(x[i, :])

        # calculate the spearman correlation between the cdfs of each variable
        sp_corr = scipy.stats.spearmanr(cdfs, axis=1)[0]
        assert sp_corr.shape[0] == cdfs.shape[0], "The size of correlation matrix doesn't match the number of stocks"

        # examine sp_corr is PSD
        if not is_psd(sp_corr):
            raise ValueError
        self.spearmanr = sp_corr

    def fit(self, x, fitters):
        """
        Find the distributions of each varaibles.

        params:
            - x: a 2D numpy arrray, each row represents a variable
            - fitters: a list of DistFitters to fit each variable into a distribution
        """
        assert x.shape[0] == len(fitters), "Each variable should has its own fitter"

        dists = []
        # fit data into distributions
        for i in range(x.shape[0]):
            fitters[i].fit(x[i, :])
            dists.append(fitters[i].fitted_dist)
        self.dists = dists
        self._update_spearmanr(x)

    def simulate(self, nsample):
        simulator = CholeskySimulator(self.spearmanr)
        std_norm_vals = simulator.simulate(nsample)
        std_norm_cdfs = scipy.stats.norm(loc=0, scale=1).cdf(std_norm_vals)

        # for each row in standard normal cdfs, reverse them 
        # to the actual value using correspondent distributions  
        sim_vals = np.empty(shape=std_norm_cdfs.shape, dtype=float)
        for i in range(sim_vals.shape[0]):
            sim_vals[i, :] = self.dists[i].ppf(std_norm_cdfs[i, :])
        return sim_vals