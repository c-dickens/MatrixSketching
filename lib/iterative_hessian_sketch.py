import numpy as np 
from timeit import default_timer as timer
from gaussian_sketch import GaussianSketch
from count_sketch import CountSketch
from sparse_jlt import SparseJLT
from srht_sketch import SRHTSketch


class IterativeHessianOLS:
    """
    An iterative solver for the optimisation problem
    f(x) = 1/2 ||Ax - y||_2^2
    by randomised newton method.
    """
    
    def __init__(self,n_data_rows:int, n_data_cols:int,\
                sk_dim:int,sk_mode='Gaussian',sparse_data=None,\
                ihs_mode='multi',sjlt_sparsity=5):
        """
        Instantiates the IHS sketching construction.
        """
        self.sk_mode      = sk_mode
        self.n_data_rows  = n_data_rows
        self.n_data_cols  = n_data_cols
        self.sk_dim       = min([sk_dim,n_data_rows])
        self.ihs_mode     = ihs_mode

        if self.sk_mode == 'Gaussian':
            self.sketcher = GaussianSketch(self.sk_dim,self.n_data_rows,self.n_data_cols)
        elif self.sk_mode == 'SRHT':
            # Add 1 to the number of data columns as we append a column for y later on
            try: 
                self.sketcher = SRHTSketch(self.sk_dim,self.n_data_rows,self.n_data_cols,'HAD')
            except:
                self.sketcher = SRHTSketch(self.sk_dim,self.n_data_rows,self.n_data_cols,'DCT')
        elif self.sk_mode == 'SJLT':
            self.sketcher = SparseJLT(self.sk_dim,self.n_data_rows,self.n_data_cols,col_sparsity=sjlt_sparsity)
        elif self.sk_mode == 'CountSketch':
            self.sketcher = CountSketch(self.sk_dim,self.n_data_rows,self.n_data_cols)
        
        self.sparse_data_is_set = False # Init for dense sketches to False
        if (sparse_data is not None) and (self.sk_mode == 'SJLT' or self.sk_mode == 'CountSketch'):
            self.sketcher.set_sparse_data(sparse_data)
            self.sparse_data_is_set = True

    def _init_iterations(self,X,y,iterations):
        """
        Initialises the arrays we use for iterations
        - current_weights the vector we will update under iterative scheme
        - weights_hist is an array which contains all of the updated weights used for error history
        - XTy is the projection of the targets onto the column space of the data X
        """
        current_weights = np.zeros((self.n_data_cols,1),dtype=float)
        weights_hist = np.zeros((self.n_data_cols,iterations),dtype=float)
        XTy = (X.T@y).reshape(-1,1)
        return current_weights, weights_hist, XTy


    def _grad(self, X, vec, XTy):
        """
        Returns the gradient function 
        nabla f(x) = A.T ( Ax - y )

        We use as input the vector XTy so no need to recompute
        """
        return X.T@(X@vec) - XTy

    def _iterate_multiple(self,X,y,iterations=10,timing=False):
        if timing:
            return self._iterate_multiple_timing(X,y,iterations)
        current_x, all_x, XTy = self._init_iterations(X,y,iterations)
        for it in range(iterations):
            #######################################################
            # 1. Generate a sketch and obtain the svd factors for efficient solving.
            self.sketcher.sketch(X,seed=1000*it)
            u,sig,vt = self.sketcher.get(in_svd=True)
            sig = sig[:,np.newaxis]
            sig_inv = 1./sig
            gradient = self._grad(X, current_x, XTy)
            update = - vt.T@ (sig_inv**2 * (vt @ gradient)) # This solves lineat system H update = - gradient
            current_x += update
            all_x[:,it] = current_x[:,0]
            #######################################################
            # print(u.shape, sig.shape,vt.shape)
            # SA = self.sketcher.get(in_svd=False)
            # H = SA.T @ SA
            # H_inv = np.linalg.pinv(H)

            # 2. Solve linear system wrt negative gradient
            # Equivalent to Hessian @ new weights = - gradient f(old weights)
            # right_hand_side = - self._grad(X, current_x, XTy)
            # print('Grad shape: ', right_hand_side.shape)
            # #update = (vt.T@(sig_inv*u.T))@right_hand_side
            # print('Basis shape: ', (vt @ right_hand_side).shape)
            # print('Term shape: ', (sig_inv**2 * (vt @ right_hand_side)).shape)
            # update =  - vt.T@(sig_inv**2 * (vt @ right_hand_side))
            
            #print(H.shape, self._grad(X, current_x, XTy).shape)
            # current_x = current_x - H_inv @ self._grad(X, current_x, XTy) # ! This works
            # current_x = current_x - vt.T@ (sig_inv**2 * (vt @ self._grad(X, current_x, XTy)))  # ! This works
            # update = - vt.T@ (sig_inv**2 * (vt @ self._grad(X, current_x, XTy)))
            # current_x += update # ! This works
            # all_x[:,it] = current_x[:,0]
        return current_x, all_x

    def _iterate_multiple_timing(self,X,y,iterations=10):
        """
        Performs the iterations but also records the timing of each individual part.
        """
        times = {
        'Total'    : 0.,
        'Sketch'   : np.zeros(iterations,dtype=float),
        'SVD'      : np.zeros(iterations,dtype=float),
        'Solve'    : np.zeros(iterations,dtype=float)
        }
        TIMER_START = timer()
        current_x, all_x, XTy = self._init_iterations(X,y,iterations)
        for it in range(iterations):
            #######################################################
            # 1. Generate a sketch and obtain the svd factors for efficient solving.
            SKETCH_TIMER = timer()
            self.sketcher.sketch(X,seed=1000*it)
            times['Sketch'][it] = timer() - SKETCH_TIMER

            SVD_TIMER = timer()
            u,sig,vt = self.sketcher.get(in_svd=True)
            times['SVD'][it] = timer() - SVD_TIMER
            sig = sig[:,np.newaxis]
            sig_inv = 1./sig

            SOLVE_TIME = timer()
            gradient = self._grad(X, current_x, XTy)
            update = - vt.T@ (sig_inv**2 * (vt @ gradient)) # This solves lineat system H update = - gradient
            current_x += update
            times['Solve'][it] = timer() - SOLVE_TIME
            all_x[:,it] = current_x[:,0]
            #######################################################
        times['Total']= timer() - TIMER_START
        return current_x, all_x,times

    def fit(self,X,y,iterations=10,timing=False):
        """
        Fits the model without any timing on data X and targets y
        """
        if self.ihs_mode == 'multi':
            if timing:
                return self._iterate_multiple(X,y,iterations,timing=True)
            x, all_x  = self._iterate_multiple(X,y,iterations)
        return x, all_x


        