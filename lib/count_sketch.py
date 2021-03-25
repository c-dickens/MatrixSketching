import numpy as np
from numba import njit
from matrix_sketch import MatrixSketch
from sparse_data_converter import SparseDataConverter
from scipy import sparse
from scipy.sparse import coo_matrix

@njit(fastmath=True)
def fast_countSketch(SA,row,col,data,sign,row_map):
    for idx in range(len(data)):
        SA[row_map[row[idx]],col[idx]]+=data[idx]*sign[row[idx]]
    return SA

class CountSketch(MatrixSketch):
    """
    Class wrapper for the CountSketch of Clarkson and Woodruff
    """
    def __init__(self,sketch_dim:int, n_data_rows:int, \
        n_data_cols:int):
        """
        # TODO: docstring
        ...
        """
        super(CountSketch,self).__init__(sketch_dim)
        super(CountSketch,self)._prepare_sketch(n_data_rows, n_data_cols)


    def get_sparse_data(self,mat):
        """
        Sets class variables so the sketch can access the data again when necessary.

        self.sparse_data_flag -- Bool that indicates 
        """
        sparse_data = SparseDataConverter(mat)
        self.rows = sparse_data.rows
        self.cols = sparse_data.cols
        self.vals = sparse_data.vals
        self.sparse_data_flag = True
    
    def set_sparse_data(self,mat):
        """
        if mat is a scipy.sparse.coo_matrix then just set the variables
        """
        assert isinstance(mat, sparse.coo.coo_matrix)
        self.rows = mat.row
        self.cols = mat.col
        self.vals = mat.data
        self.sparse_data_flag = True

    def _sample_hashes(self,seed=100):
        """
        Samples the random linear transform S with 'seed'
        S is a Gaussian random transform.
        """
        np.random.seed(seed)
        row_map = np.random.choice(self.sketch_dim,self.n_data_rows,replace=True)
        sign_map = np.random.choice([-1.,1.], self.n_data_rows, replace=True)
        return row_map, sign_map

    def sketch(self,mat,seed=100):
        """
        Performs the sketch

        # TODO insert a check on the matrix being sketched
        """
        if not self.sparse_data_flag:
            self.get_sparse_data(mat)
        self.sketch_matrix = np.zeros((self.sketch_dim,self.n_data_cols),dtype=float)
        buckets,signs = self._sample_hashes(seed)
        self.sketch_matrix = fast_countSketch(self.sketch_matrix,\
            self.rows,\
            self.cols,\
            self.vals,\
            signs,\
            buckets)

        
