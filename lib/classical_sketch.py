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
            self.sketcher = SparseJLT(self.sk_dim,self.n_data_rows,self.n_data_cols,col_sparsity=5)
        elif self.sk_mode == 'CountSketch':
            self.sketcher = CountSketch(self.sk_dim,self.n_data_rows,self.n_data_cols)
        
        if (sparse_data is not None) and (self.sk_mode == 'SJLT' or self.sk_mode == 'CountSketch'):
            self.sketcher.set_sparse_data(sparse_data)
            print('The sparse data has been set')
            print(len(self.sketcher.rows))

    def _sketch(self,seed):
        pass
    
    def _solve(self):
        pass

    def fit(self,X,y,seed=100):
        """
        Fits the sketched regression model with a classical sketch to 
        data X and y.
        """
        np.random.seed(seed)
        Xy = np.c_[X,y]
        self.sketcher.sketch(Xy,seed)
        _sketch = self.sketcher.get()
        SX, Sy = _sketch[:,:-1], _sketch[:,-1]
        x = np.linalg.lstsq(SX, Sy,rcond=None)[0]
        return x



        