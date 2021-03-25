import numpy as np
from scipy import sparse
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix

class SparseDataConverter:
    """
    Moves between sparse and dense data
    """

    def __init__(self, mat):
        """
        converts mat to sparse data
        """
        # Convert data into necessary format
        # LOGIC: if self.data is sparse just make references for later
        # otherwise, convert to sparse data.
        if isinstance(mat, sparse.coo.coo_matrix):
            self.coo_data = self.data
            self.rows = self.coo_data.row
            self.cols = self.coo_data.col
            self.vals = self.coo_data.data
        elif isinstance(mat, sparse.csr.csr_matrix) or isinstance(mat, sparse.csc.csc_matrix):
            self.coo_data = self.data.tocoo()
            self.rows = self.coo_data.row
            self.cols = self.coo_data.col
            self.vals = self.coo_data.data
        else:
            # Is numpy ndarray
            self.coo_data = coo_matrix(mat)
            self.rows = self.coo_data.row
            self.cols = self.coo_data.col
            self.vals = self.coo_data.data
