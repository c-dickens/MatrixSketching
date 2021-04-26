"""
A collection of useful functions for the experiments
"""
import numpy as np
from math import exp, floor
from scipy.fftpack import dct


# * Common sketching paradigms
models = ['IHS', 'Classical']
methods = ['CountSketch', 'SRHT', 'SJLT', 'Gaussian']

def experimental_data(n,d, sigma=1.0,seed=100):
    """
    The data for this experiment is taken from 3.1 https://jmlr.org/papers/volume17/14-460/14-460.pdf
    A       : n \times d matrix of N(0,1) entries
    x_model : d \times 1 vector chosen uniformly from the d dimensional sphere by choosing 
              random normals and then normalising.
    w       : Vector of noise perturbation which is N(0,sigma**2*I_n)
    y = A@x_model + w
    """
    np.random.seed(seed)
    A = np.random.randn(n,d)
    x_model = np.random.randn(d,1)
    x_model /= np.linalg.norm(x_model,axis=0)
    w = np.random.normal(loc=0.0,scale=sigma,size=(n,1))
    #w = sigma**2*np.random.randn(n,1)
    y = A@x_model + w
    return y,A,x_model

def shi_phillips_ridge_data(n,d,effective_rank,tail_strength=None,seed=100):
    '''
    Generates the low rank data from the shi-phillips paper.
    ''' 
    A = np.zeros((n,d),dtype=float)
    if effective_rank is None:
        effective_rank = 0.25 
    r = effective_rank
    for col in range(d):
        arg = (col)**2/ r**2
        std = exp(-arg) 
        A[:,col] = np.random.normal(loc=0.0,scale=std**2,size=(n,))
    x_star = np.zeros(d)
    x_star[:r] = np.random.randn(r,)
    x_star /= np.linalg.norm(x_star,2)
    noise = np.random.normal(loc=0.0,scale=4.0,size=(n,))
    X = dct(A)
    y = X@x_star + noise
    return X, y, x_star

def svd_solve(a,b):
    u,sig,vt = np.linalg.svd(a,full_matrices=False)
    v = vt.T
    #safe_sig = sig[sig > 1E-8]
    #d_safe = len(safe_sig)
    #u = u[:,:d_safe]
    #v = v[:,:d_safe]
    sig = sig[:,np.newaxis]
    #sig = safe_sig[:,np.newaxis]
    #print(sig)
    sig_inv = 1./sig
    x_opt = (v@(sig_inv*u.T))@b
    # ! Beware of multindexed arrays which can affect error comparison
    # ! x_opt = x_opt[:,0]
    return x_opt

def svd_ridge_solve(X,y,gamma):
    u,sig,vt = np.linalg.svd(X,full_matrices=False)
    v = vt.T
    sig = sig[:,np.newaxis]
    diag_scaler = sig / (sig**2 + gamma)
    x_opt = v@(diag_scaler * (u.T @ y))
    return x_opt

def sparsify_data(mat, sparsity=0.2,seed=100):
    """
    Randomly zeroes out some coordinates of the input matrix mat
    Only operates on numpy arrays.
    """
    np.random.seed(seed)
    n,d = mat.shape
    mat_sparse = np.zeros_like(mat,dtype=mat.dtype)
    num_nnzs = int(sparsity*n*d)
    nnz_row_locs = np.random.choice(n,size=(num_nnzs,),replace=True)
    nnz_col_locs = np.random.choice(d,size=(num_nnzs,),replace=True)
    mat_sparse[nnz_row_locs, nnz_col_locs] = mat[nnz_row_locs, nnz_col_locs]
    return mat_sparse


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

def test_mse(test_mat,weights,test_targets):
    """
    Returns the testing error 1/n_test * ||y_test - test_mat * weights||^2
    for a given set of weights
    """
    return np.sqrt(1./test_mat.shape[0] * np.linalg.norm(test_targets - test_mat @ weights)**2)