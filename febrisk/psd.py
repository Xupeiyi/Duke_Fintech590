import numpy as np

from febrisk.statistics import frobenius_norm


def is_psd(matrix, tolerance=1e-8):
    """
    Examine if matrix is PSD by checking if its eigen values are all non-negative.
    
    params:
        - matrix: a real symmetric matrix.
        - tolerance
    """
    # examine if a matrix is real symmetric
    if abs(matrix - matrix.T).sum() > tolerance:
        raise ValueError("This function is for real symmetric matrices!")
    
    eig_values, _ = np.linalg.eigh(matrix)
    return all(eig_values > -tolerance)


# ================================
# Rebonato and Jackel's Near PSD 
# ================================

def near_psd(corr):
    """
    Rebonato and Jackel's method for finding an acceptable PSD matrix.
    corr: a correlation matrix.
    """
    # update the eigen value and scale
    eig_vals, eig_vecs = np.linalg.eigh(corr)
    eig_vals[eig_vals < 0] = 0
    
    ts = 1 / (np.square(eig_vecs) @ eig_vals)
    sqrt_T = np.diagflat(np.sqrt(ts))
    sqrt_lambda = np.diag(np.sqrt(eig_vals))
    
    root = sqrt_T @ eig_vecs @ sqrt_lambda # B = sqrt(𝑇) * S * sqrt(Λ')
    near_corr = root @ root.T

    return near_corr

# ===================================
# Higham's Nearest PSD
# ===================================

def projection_u(matrix):
    """Projection U sets diagonal elements to 1."""
    new_matrix = matrix.copy()
    np.fill_diagonal(new_matrix, 1)
    return new_matrix


def projection_s(matrix):
    """
    Projection S reconstructs a matrix by setting the 
    negative eigen values of the original matrix to zero.
    """
    eig_vals, eig_vecs = np.linalg.eigh(matrix)
    eig_vals[eig_vals < 0] = 0
    return eig_vecs @ np.diag(eig_vals) @eig_vecs.T


def nearest_psd(corr, max_iter=100, tolerance=1e-9):
    """
    Use Higham's method to generate the nearest PSD of a correlation matrix.
    There is no gaurantee that this function can generate a PSD.

    params:
        - corr: the correlation matrix.
        - max_iter: maximum number of iterations.
        - tolerance: break the iteration if we are not able to improving 
                     the frobenius norm by at least tolerance value.
    """
    # ΔS0 = 0, Y0 = A, γ0 = max float
    delta_s = 0
    y = corr
    prev_gamma = np.finfo(np.float64).max

    # Loop k ∈ 1... max Iterations
    for i in range(max_iter):
        r = y - delta_s          # Rk = Yk-1 − ΔSk-1
        x = projection_s(r)      # Xk = Ps(Rk)
        delta_s = x - r          # ΔSk = Xk − Rk
        y = projection_u(x)      # Yk = Pu(Xk)
        gamma = frobenius_norm(y - corr)
        
        # if |γk-1 − γk | < tol then break
        if abs(gamma - prev_gamma) < tolerance:  
            break
        prev_gamma = gamma
    
    return y
