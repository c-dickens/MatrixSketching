"""
A collection of useful functions for the experiments
"""
import numpy as np

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