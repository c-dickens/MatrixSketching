import numpy as np 
from gaussian_sketch import GaussianSketch 
from srht_sketch import SRHTSketch


class ClassicalSketch:
    """
    """
    def __init__(self, n_data_rows:int, n_data_cols:int,\
                sk_dim:int,sk_mode='Gaussian'):
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
            self.sketcher = SRHTSketch(self.sk_dim,self.n_data_rows,self.n_data_cols+1,'HAD')

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
        x = np.linalg.lstsq(SX, Sy)[0]
        return x



        