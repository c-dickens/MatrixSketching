import numpy as np 
from gaussian_sketch import GaussianSketch
from count_sketch import CountSketch
from sparse_jlt import SparseJLT
from srht_sketch import SRHTSketch


class ClassicalSketch:
    """
    """
    def __init__(self,n_data_rows:int, n_data_cols:int,\
                sk_dim:int,sk_mode='Gaussian',sparse_data=None):
        """
        Approximate OLS regression using random projections
        Parameters:
        - rp_dim (int)    : the number of rows retained in the random projection.
        - rp_mode (str)   : sketch mode used to decide on the sketch.
            - method: sketch and solve or iterative hessian
        """
        self.sk_mode      = sk_mode
        self.n_data_rows  = n_data_rows
        self.n_data_cols  = n_data_cols
        self.sk_dim       = min([sk_dim,n_data_rows])
        if self.sk_mode == 'Gaussian':
            self.sketcher = GaussianSketch(self.sk_dim,self.n_data_rows,self.n_data_cols)
        elif self.sk_mode == 'SRHT':
            # Add 1 to the number of data columns as we append a column for y later on
            try: 
                self.sketcher = SRHTSketch(self.sk_dim,self.n_data_rows,self.n_data_cols+1,'HAD')
            except:
                self.sketcher = SRHTSketch(self.sk_dim,self.n_data_rows,self.n_data_cols+1,'DCT')
        elif self.sk_mode == 'SJLT':
            self.sketcher = SparseJLT(self.sk_dim,self.n_data_rows,self.n_data_cols+1,col_sparsity=5)
        elif self.sk_mode == 'CountSketch':
            self.sketcher = CountSketch(self.sk_dim,self.n_data_rows,self.n_data_cols+1)
        
        if (sparse_data is not None) and (self.sk_mode == 'SJLT' or self.sk_mode == 'CountSketch'):
            self.sketcher.set_sparse_data(sparse_data)
            self.sparse_data_is_set = True

    def _sketch_data_targets(self,X,y,seed=100):
        """
        Sketches the [data,target] concatenated data for sketch-and-solve.
        Slightly different functionality for dense or sparse sketches as sparse 
        data is set in the __init__ function 
        """
        if self.sparse_data_is_set:
            self.sketcher.sketch(seed)
        else:
            Xy = np.c_[X,y]
            self.sketcher.sketch(Xy,seed)
       
    def _solve(self,in_svd=False):
        """
        Gets the sketch matrix and splits into the [SX, Sy] parts.
        Then solves the regression instance using SVD
        """
        _sketch = self.sketcher.get(in_svd=False)
        SX, Sy = _sketch[:,:-1], _sketch[:,-1].reshape(-1,1)
        u,sig,vt = np.linalg.svd(SX,full_matrices=False)
        sig = sig[:,np.newaxis]
        sig_inv = 1./sig
        weights = (vt.T@(sig_inv*u.T))@Sy
        weights = np.linalg.lstsq(SX, Sy,rcond=None)[0] # rcond = None is to ignore a warning flag.
        return weights

    def fit(self,X,y,seed=100,in_svd=False):
        """
        Fits the sketched regression model with a classical sketch to 
        data X and y.
        First step is to sketch the data using the given sketch from init.
        Second step is to solve the regression instance, using either the lstsq
        solver, or in SVD format.
        """
        self._sketch_data_targets(X,y,seed)
        weights = self._solve()
        return weights
       



        