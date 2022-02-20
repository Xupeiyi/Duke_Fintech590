from bisect import bisect_left

import scipy
import numpy as np

from febrisk.stats import PCA


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
        L = self.factorize(explained, verbose)
        Z = scipy.random.randn(L.shape[1], nsample)
        return L @ Z
