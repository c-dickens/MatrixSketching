import numpy as np 

class MatrixSketch:
    """
    A generic class which calls different matrix sketches.

    Applies a sketch to matrix A of size n \times d to generate 
    a sketch B of size m \times d.
    """

    def __init__(self,sketch_dim:int):
        """
        Inputs: 
            sketch_dim - the number of rows to project down to for B
        """
        self.sketch_dim = sketch_dim
        print('Init MatrixSketch')
        print('Sketch Dim:', self.sketch_dim)

    def _prepare_sketch(self,n_data_rows:int, n_data_cols:int):
        """
        Helper function to set parameters of the sketch and data we may need
        """
        self.n_data_rows = n_data_rows
        self.n_data_cols = n_data_cols

    def get(self,in_svd=False):
        """
        Returns the sketch self.sketch_matrix either in full form or in SVD form
        """
        if in_svd:
            u,s,vt = np.linalg.svd(self.sketch_matrix,full_matrices=False)
            return u,s,vt
        else:
            return self.sketch_matrix
