"""
A collection of useful functions for the experiments
"""
import numpy as np


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

def test_mse(test_mat,weights,test_targets):
    """
    Returns the testing error 1/n_test * ||y_test - test_mat * weights||^2
    for a given set of weights
    """
    return 1./test_mat.shape[0] * np.linalg.norm(test_targets - test_mat @ weights)**2