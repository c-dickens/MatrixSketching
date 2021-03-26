"""
A collection of useful functions for the experiments
"""
import numpy as np

def prediction_error(mat,vec1,vec2):
    return (1./len(mat))*np.linalg.norm(mat@(vec1 - vec2),ord=2)**2