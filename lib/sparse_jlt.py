import numpy as np
from math import sqrt as m_sqrt
from numba import njit
from matrix_sketch import MatrixSketch
from count_sketch import CountSketch, fast_countSketch
from sparse_data_converter import SparseDataConverter
from scipy import sparse
from scipy.sparse import coo_matrix

class SparseJLT(CountSketch):
    """
    Class wrapper for the Sparse Johnson Lindesntrauss Transform of Kane and Nelson


    1. Generate `s` independent countsketches each of size m/s x n and concatenate them.
    2. Use initial hash functions as decided above in the class definition and then generate new hashes for
    subsequent countsketch calls.
    """
    def __init__(self,sketch_dim:int, n_data_rows:int, \
        n_data_cols:int,col_sparsity:int):
        """
        # TODO: docstring
        ...
        """
        super(SparseJLT,self).__init__(sketch_dim,n_data_rows, n_data_cols)
        self.col_sparsity = col_sparsity
        assert self.col_sparsity > 1

    def _sample_hashes(self,seed=100):
        """
        returns the row and sign hash functions for SJLT
        """
        # ! NB row map is used for indexing so MUST be an int!!!
        row_map = np.zeros((self.col_sparsity,self.n_data_rows),dtype=int)
        sign_map = np.zeros((self.col_sparsity,self.n_data_rows))
        for _ in range(self.col_sparsity):
            np.random.seed(seed + _)
            row_map[_,:] = np.random.choice(self.sjlt_proj_dim,self.n_data_rows,replace=True)
            sign_map[_,:] = np.random.choice([-1.,1.], self.n_data_rows, replace=True) 
        return row_map, sign_map


    def sketch(self,mat,seed=100):
        """
        # TODO - Insert Doc string
        """
        if not self.sparse_data_flag:
            self.get_sparse_data(mat)
        self.sketch_matrix = np.zeros((self.sketch_dim,self.n_data_cols),dtype=float)
        self.sjlt_proj_dim = self.sketch_dim // self.col_sparsity
        local_summary = np.zeros((self.sjlt_proj_dim,self.n_data_cols),dtype=float)
        rows,signs = self._sample_hashes(seed)

        for batch in range(self.col_sparsity):
            rmap = rows[batch,:]
            smap = signs[batch,:]
            B = fast_countSketch(np.zeros((self.sjlt_proj_dim,self.n_data_cols),dtype=float),
                    self.rows,
                    self.cols,
                    self.vals,
                    smap,
                    rmap)
            
            id_start, id_end = batch*self.sjlt_proj_dim,(batch+1)*self.sjlt_proj_dim
            self.sketch_matrix[id_start:id_end,:] = B
        self.sketch_matrix /= m_sqrt(self.col_sparsity)

