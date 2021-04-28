import numpy as np
import sys
import os
from timeit import default_timer as timer
from gaussian_sketch import GaussianSketch
from count_sketch import CountSketch
from sparse_jlt import SparseJLT
from srht_sketch import SRHTSketch
# from pathlib import Path
# path = Path(os.getcwd())
# sys.path.append(str(path.parent.parent/ 'src'))
from frequent_directions import FastFrequentDirections, RobustFrequentDirections

class IterativeRidge:
    """
    An iterative solver for the optimisation problem
    f(x) = 1/2 ||Ax - y||_2^2 + gamma/2 ||x||_2^2
    by deterministic/randomised newton method.
    """
    def __init__(self,n_data_rows:int, n_data_cols:int,\
            sk_dim:int,sk_mode='Gaussian',sparse_data=None,\
            ihs_mode='single',sjlt_sparsity=5,gamma=1.0,batch_size=None):

    
        """
        #def __init__(self, sk_dim:int,sk_mode='FD',gamma=1.0,batch_size=None):
        Approximate ridge regression using the FD sketch.
        sk_dim (int) - the number of rows retained in the FD sketch.
        sk_mode (str) : mode for frequent directions FD or RFD.
        alpha : float - the regularisation parameter for ridge regression.
        """
        self.gamma        = gamma
        self.sk_mode      = sk_mode
        self.n_data_rows  = n_data_rows
        self.n_data_cols  = n_data_cols
        self.sk_dim       = min([sk_dim,n_data_cols]) #sk_dim
        self.ihs_mode     = ihs_mode
        # if self.sk_mode not in ['FD', 'RFD']:
        #     raise NotImplementedError('Only F(ast) and R(obust) FD methods are supported.')
       
        if self.sk_mode == 'Gaussian':
            self.sketcher = GaussianSketch(self.sk_dim,self.n_data_rows,self.n_data_cols)
        elif self.sk_mode == 'SRHT':
            try: 
                self.sketcher = SRHTSketch(self.sk_dim,self.n_data_rows,self.n_data_cols,'HAD')
            except:
                self.sketcher = SRHTSketch(self.sk_dim,self.n_data_rows,self.n_data_cols,'DCT')
        elif self.sk_mode == 'SJLT':
            self.sketcher = SparseJLT(self.sk_dim,self.n_data_rows,self.n_data_cols,col_sparsity=sjlt_sparsity)
        elif self.sk_mode == 'CountSketch':
            self.sketcher = CountSketch(self.sk_dim,self.n_data_rows,self.n_data_cols)
        # Set ihs mode to be single for deterministic sketches
        elif self.sk_mode == 'FD':
            self.ihs_mode = 'single'
            self.sketcher = FastFrequentDirections(self.sk_dim,self.n_data_rows,self.n_data_cols)
        elif self.sk_mode == 'RFD':
            self.ihs_mode = 'single'
            self.sketcher = RobustFrequentDirections(self.sk_dim,self.n_data_rows,self.n_data_cols)

        if batch_size == None:
            self.batch_size = self.sk_dim
        else:
            self.batch_size = batch_size

        self.sparse_data_is_set = False # Init for dense sketches to False
        if (sparse_data is not None) and (self.sk_mode == 'SJLT' or self.sk_mode == 'CountSketch'):
            self.sketcher.set_sparse_data(sparse_data)
            self.sparse_data_is_set = True


    def fast_iterate_single(self,X,y,iterations,seed=100):
        """
        Performs the iterations of ifdrr efficiently in small space and time.

        NB. The sketch time include sketch + SVD for randomised method.
        update time returns Add time
        """

        # * Initialisation not timed
        d = X.shape[1]
        w = np.zeros((d,1),dtype=float)
        all_w = np.zeros((d,iterations))
        measurables = {
        'sketch time' : None,
        'update time' : np.zeros(iterations,dtype=float),
        'all_times'   : np.zeros(iterations+1,dtype=float),
        'total'       : None
        # 'sketch'      : None
        }

        # ! Sketching
        if self.sk_mode in ['FD', 'RFD']:
            TIMER_START = timer()
            self.sketcher.sketch(X,batch_size=self.batch_size)
            _, SigSq, Vt, implicit_reg = self.sketcher.get_fd_outputs()
            # * NB. If d not large enough then implicit reg will be small.
            # print('Implicit reg: ', implicit_reg)
        else:
            # Note that the SVD here causes an increase in the sketch time complexity.
            TIMER_START = timer()
            self.sketcher.sketch(X,seed=109*seed)
            _, sig, Vt = self.sketcher.get(in_svd=True)
            SigSq = sig**2
            implicit_reg = 0.0 # the randomised methods have no implicit reg parameters
        SKETCH_TIME = timer() - TIMER_START

        # Extra parameters we may need 
        XTy = (X.T@y).reshape(-1,1)
        V = Vt.T
        invTerm = (1./(SigSq + implicit_reg + self.gamma )).reshape(-1,1)

        # * This lambda function evaluates H^{-1}g efficiently for gradient vector g
        H_inv_grad = lambda g, vtg : (1/self.gamma )*(g - V@vtg) + V@(invTerm*vtg)
        ridge_grad = lambda w : X.T@(X@w) + self.gamma*w - XTy

        for it in range(iterations):  
            UPDATE_TIMER_START = timer() 
            grad = ridge_grad(w) #X.T@(X@w) + self.gamma*w - XTy
            VTg = Vt@grad
            update = H_inv_grad(grad, VTg)
            w += - update
            measurables['update time'][it] = timer() - UPDATE_TIMER_START
            measurables['all_times'][it+1] = timer() - TIMER_START
            all_w[:,it] = np.squeeze(w)
        measurables['sketch time'] = SKETCH_TIME
        measurables['total'] = timer() - TIMER_START
        return w, all_w, measurables

    def _init_iterations_multi(self,X,y,iterations):
        """
        Initialises the arrays we use for iterations
        - current_weights the vector we will update under iterative scheme
        - weights_hist is an array which contains all of the updated weights used for error history
        - XTy is the projection of the targets onto the column space of the data X
        """
        current_weights = np.zeros((self.n_data_cols,1),dtype=float)
        weights_hist = np.zeros((self.n_data_cols,iterations),dtype=float)
        measurables = {
        'sketch time' : np.zeros(iterations,dtype=float),
        'update time' : np.zeros(iterations,dtype=float),
        'all_times'   : np.zeros(iterations+1,dtype=float),
        'SVD'         : np.zeros(iterations,dtype=float),
        'Solve'       : np.zeros(iterations,dtype=float),
        }
        XTy = (X.T@y).reshape(-1,1)
        return current_weights, weights_hist,measurables, XTy

    def fast_iterate_multi(self,X,y,iterations,seed=100):
        """
        Performs the iterations of ifdrr efficiently in small space and time.


        NB. The update time returns SVD + Add time
        """

        # * Initialisation not timed
        w,all_w,measurables,XTy = self._init_iterations_multi(X,y,iterations)
 

        # ! Sketching
        if self.sk_mode in ['FD', 'RFD']:
            raise NotImplementedError('F(ast) and R(obust) FD methods are supported only for ihs_mode = single.')

        # * This lambda function evaluates H^{-1}g efficiently for gradient vector g
        H_inv_grad = lambda g, vtg, V : (1/self.gamma )*(g - V@vtg) + V@(invTerm*vtg)
        ridge_grad = lambda w : X.T@(X@w) + self.gamma*w - XTy

        TIMER_START = timer()
        for it in range(iterations):   
            SKETCH_TIMER = timer()
            self.sketcher.sketch(X,seed=1000*it)
            measurables['sketch time'][it] = timer() - SKETCH_TIMER

            UPDATE_TIMER_START = timer()
            u,sig,Vt = self.sketcher.get(in_svd=True)
            sig = sig[:,np.newaxis]
            SigSq = sig**2
            invTerm = (1./(SigSq + self.gamma )).reshape(-1,1)

            grad = ridge_grad(w)
            VTg = Vt@grad
            update = H_inv_grad(grad, VTg, Vt.T)
            w += - update
            measurables['update time'][it] = timer() - UPDATE_TIMER_START
            measurables['all_times'][it+1] = timer() - TIMER_START
            all_w[:,it] = np.squeeze(w)
        measurables['total']= timer() - TIMER_START
        return w, all_w, measurables

    def fit(self,X,y,iterations=10,timing=False,seed=9999):
        if self.ihs_mode == 'single':
            return self.fast_iterate_single(X,y,iterations,seed)
        else:
            return self.fast_iterate_multi(X,y,iterations)

            


# import numpy as np 
# from timeit import default_timer as timer
# from gaussian_sketch import GaussianSketch
# from count_sketch import CountSketch
# from sparse_jlt import SparseJLT
# from srht_sketch import SRHTSketch
# from frequent_directions import FrequentDirections, FastFrequentDirections, RobustFrequentDirections



    
    # def __init__(self,n_data_rows:int, n_data_cols:int,\
    #             sk_dim:int,sk_mode='Gaussian',sparse_data=None,\
    #             ihs_mode='single',sjlt_sparsity=5,gamma=1.0,batch_size=None):
#         """
#         Instantiates the iterative ridge sketching construction.

#         In this setting we do not permit self.sk_dim to exceed the number of columns
#         """    
#         self.sk_mode      = sk_mode
#         self.n_data_rows  = n_data_rows
#         self.n_data_cols  = n_data_cols
#         self.sk_dim       = min([sk_dim,n_data_cols])
#         self.ihs_mode     = ihs_mode
#         self.gamma        = gamma






#     def _init_iterations(self,X,y,iterations):
#         """
#         Initialises the arrays we use for iterations
#         - current_weights the vector we will update under iterative scheme
#         - weights_hist is an array which contains all of the updated weights used for error history
#         - XTy is the projection of the targets onto the column space of the data X
#         """
#         current_weights = np.zeros((self.n_data_cols,1),dtype=float)
#         weights_hist = np.zeros((self.n_data_cols,iterations),dtype=float)
#         XTy = (X.T@y).reshape(-1,1)
#         return current_weights, weights_hist, XTy
        
    # def _grad(self, X, vec, XTy):
    #     """
    #     Returns the gradient function 
    #     nabla f(x) = (X.T X + gamma I)w - XTy
    #     We use as input the vector XTy so no need to recompute
    #     """
    #     return X.T@(X@vec) + self.gamma*vec  - XTy


#     def _iterate(self,X,y,iterations=10,seed=None,timing=False):
#         # if timing:
#         #     return self._iterate_multiple_timing(X,y,iterations)
#         current_x, all_x, XTy = self._init_iterations(X,y,iterations)
#         if self.sk_mode in ['FD', 'RFD']:
#             self.sketcher.sketch(X,batch_size=self.batch_size)    
#         else:
#             self.sketcher.sketch(X,seed=seed)
#         # u,sig,vt = self.sketcher.get(in_svd=True)
#         # sig = sig[:,np.newaxis]
#         # diag_scaler = sig / (sig**2 + self.gamma)
#         # for it in range(iterations):
#         #     #######################################################
#         #     # 1. Generate a sketch and obtain the svd factors for efficient solving.
#         #     gradient = self._grad(X, current_x, XTy)
#         #     update = - vt.T@ (diag_scaler * (vt @ gradient)) # This solves lineat system H update = - gradient
#         #     current_x += update
#         #     all_x[:,it] = current_x[:,0]
#         # return current_x, all_x
#         '''
#         Fits the iterated ridge model with FD
#         '''
#         d = X.shape[1]
#         w = np.zeros((d,1),dtype=float)
#         all_w = np.zeros((d,iterations))
#         XTy = (X.T@y).reshape(-1,1)
        
#         # Fit the FD
#         # if not self.is_fitted:
#         #     self._sketch(X)
#         H = self.sketcher.sketch_matrix.T@self.sketcher.sketch_matrix + (self.gamma)*np.eye(d)
#         H_inv = np.linalg.pinv(H)
#         for it in range(iterations):
#             grad = X.T@(X@w) + self.gamma*w - XTy
#             w += - H_inv@grad
#             all_w[:,it] = np.squeeze(w)
#         return np.squeeze(w), all_w

#     def fit(self,X,y,iterations=10,timing=False):
#         """
#         Fits the model without any timing on data X and targets y
#         """
#         x,all_x = self._iterate(X,y,iterations)
#         # if self.ihs_mode == 'multi':
#         #     if timing:
#         #         return self._iterate_multiple(X,y,iterations,timing=True)
#         #     x, all_x  = self._iterate_multiple(X,y,iterations)
#         return x, all_x



    # def _grad(self, X, vec, XTy):
    #     """
    #     Returns the gradient function 
    #     nabla f(x) = (X.T X + gamma I)w - XTy
    #     We use as input the vector XTy so no need to recompute
    #     """
    #     return X.T@(X@vec) + self.gamma*vec  - XTy

    # def _sketch(self, X):
    #     '''
    #     Private function for calling the sketch methods
    #     '''
    #     if self.sk_mode == 'FD':
    #         sketcher = FastFrequentDirections(X.shape[1],sketch_dim=self.sk_dim)
    #     elif self.sk_mode == 'RFD':
    #         sketcher = RobustFrequentDirections(X.shape[1],sketch_dim=self.sk_dim)
    #     sketcher.fit(X,batch_size=self.batch_size)
    #     self.sketch = sketcher.sketch
    #     self.alpha = sketcher.delta # == 0 if using FastFrequentDirections so can use self.gamma + self.alpha everywhere 
    #     self.is_fitted = True
    
    # def fit(self,X,y):
    #     '''
    #     Fits the ridge regression model on data X with targets y
    #     '''
    #     d = X.shape[1]
    #     self._sketch(X)
    #     H = self.sketch.T@self.sketch + (self.gamma+self.alpha)*np.eye(d)
    #     self.coef_ = np.linalg.solve(H, X.T@y)
    #     self.H_inv = np.linalg.pinv(H)

    # def iterate(self,X,y,iterations=10,timing=False,seed=1000):
    #     current_x, all_x, XTy = self._init_iterations(X,y,iterations)
    #     if self.sk_mode in ['FD','RFD']:
    #         print('Sketching data')
    #         self.sketcher.sketch(X, batch_size=self.batch_size)
    #         _, sigma_squared, vt, delta = self.sketcher.get_fd_outputs()
    #         sig = np.sqrt(sigma_squared)
    #     else:
    #         self.sketcher.sketch(X,seed=seed*100)
    #         _,sig,vt = self.sketcher.get(in_svd=True)
    #     sig = sig[:,np.newaxis]
    #     sig_inv = 1./sig
    #     for it in range(iterations):
    #         #######################################################
    #         # 1. Generate a sketch and obtain the svd factors for efficient solving.
    #         gradient = self._grad(X, current_x, XTy)
    #         update = - vt.T@ (sig_inv**2 * (vt @ gradient)) # This solves lineat system H update = - gradient
    #         current_x += update
    #         all_x[:,it] = current_x[:,0]
    #     return current_x, all_x
