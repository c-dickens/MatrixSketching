import sys
sys.path.insert(0,'..',)
from gaussian_sketch import GaussianSketch
import numpy as np

X = np.random.randn(10,2)
g = GaussianSketch(5,X.shape[0],X.shape[1])
g.sketch_matrix(X)