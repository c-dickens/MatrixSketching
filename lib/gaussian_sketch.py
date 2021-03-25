import numpy as np
from matrix_sketch import MatrixSketch

class GaussianSketch(MatrixSketch):
    """
    Class wrapper for the Gaussian Sketch
    """
    def __init__(self,sketch_dim:int, n_data_rows:int, \
        n_data_cols:int):

        super(GaussianSketch,self).__init__(sketch_dim)
        super(GaussianSketch,self)._prepare_sketch(n_data_rows, n_data_cols)

    def _sample(self,seed=100):
        """
        Samples the random linear transform S with 'seed'
        S is a Gaussian random transform.
        """
        np.random.seed(seed)
        S = np.random.randn(self.sketch_dim, self.n_data_rows)
        S /= np.sqrt(self.sketch_dim)
        return S

    def sketch(self,mat,seed=None):
        S = self._sample(seed)
        self.sketch_matrix = S@mat
        