"""
A collection of useful functions for the experiments
"""
import numpy as np

def svd_solve(a,b):
    u,sig,vt = np.linalg.svd(a,full_matrices=False)
    v = vt.T
    sig = sig[:,np.newaxis]
    sig_inv = 1./sig
    x_opt = (v@(sig_inv*u.T))@b
    # ! Beware of multindexed arrays which can affect error comparison
    # ! x_opt = x_opt[:,0]
    return x_opt


def vec_error(vec1,vec2):
    """
    Returns ||vec1 - vec2||_2
    """
    return np.linalg.norm(vec1 - vec2)

def prediction_error(mat,vec1,vec2):
    """
    Returns the prediction semi-norm
    1/sqrt(n) * ||mat(vec1 - vec2)||_2
    """
    return 1./(np.sqrt(mat.shape[0]))*np.linalg.norm(mat@(vec1 - vec2),ord=2)