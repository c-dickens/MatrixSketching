import sys
sys.path.append('../')
import numpy as np
import pandas as pd
import itertools
from timeit import default_timer as timer
from sparse_data_converter import SparseDataConverter
from scipy import sparse
from scipy.sparse import coo_matrix
from classical_sketch import ClassicalSketch
from iterative_hessian_sketch import IterativeHessianOLS
from experiment_utils import models, methods, experimental_data,svd_solve, prediction_error, test_mse




def main(n,d,i,t):
    """
    Task: IHS-OLS with random projections, specifically, the CountSketch
    but with wall clock time to understand how the sketches fare in terms 
    of time.


    Figure 2 https://jmlr.org/papers/volume17/14-460/14-460.pdf

    Evaluates the error as a function of the iterations with respect 
    to the optimal weights.

    We use a default train/test split of 0.9 train and 0.1 test
    """
    # * Experimental setup      
    # *  2 Sketch parameters taken from IHS paper
    file_path = 'results/experiment2-ihs-iterations-time.csv' 
    #n = 6000
    #d = 200
    n_train = int(0.9*n)
    n_test = n - n_train
    num_iters = i
    num_trials = t
    ihs_sketch_dims = [5*d]
    
    # * Results setup 
    metrics = ['Sketch','SVD','Solve','Coefficient Error', 'Test Error']
    methods_and_exact = ['Exact(SVD)'] + methods
    all_setups = list(itertools.product(methods, ihs_sketch_dims)) 
    df = pd.DataFrame(np.zeros((num_iters,len(methods_and_exact)*len(metrics))))
    df.columns = pd.MultiIndex.from_product([methods_and_exact,metrics])
    ihs_results_opt = {mg: np.zeros(num_iters,dtype=float) for mg in all_setups}

    for t in range(num_trials):
        print(f'Trial {t}')
         # * 0. Data setup and sparsification
        y, A, x_model = experimental_data(n,d,1.0,seed=t)
        A_train, A_test = A[:n_train,:], A[n_train:,:]
        y_train, y_test = y[:n_train,:], y[n_train:,:]

        sparse_data = sparse.coo_matrix(A_train) 
        SVD_TIMER = timer()
        x_opt = svd_solve(A_train,y_train)
        svd_time = timer() - SVD_TIMER 
        assert x_opt.shape == x_model.shape
        df['Exact(SVD)','SVD'] += svd_time
        df['Exact(SVD)', 'Test Error'] += test_mse(A_test,x_opt,y_test)

        # ! Sparse methods using NUMBA need to compile the sketch so let's do that ahead of time
        # ! so that the timing experiment is not compromised.
        if t == 0:
            c_ihs_solver = IterativeHessianOLS(n_train,d,ihs_sketch_dims[0],'CountSketch',sparse_data=sparse_data)
            s_ihs_solver = IterativeHessianOLS(n_train,d,ihs_sketch_dims[0],'SJLT',sparse_data=sparse_data)
            ctsk_setup = c_ihs_solver.fit(A_train,y_train,num_iters,timing=True)
            sjlt_setup = s_ihs_solver.fit(A_train,y_train,num_iters,timing=True)

        # * 2b IHS
        for sk_method_gamma_d in all_setups:
            sk_method, ihs_sk_dim = sk_method_gamma_d[0], sk_method_gamma_d[1]
            ihs_solver = IterativeHessianOLS(n_train,d,ihs_sk_dim,sk_method,sparse_data=sparse_data)
            x_ihs,x_hist,ihs_total_time = ihs_solver.fit(A_train,y_train,num_iters,timing=True)
            assert x_opt.shape == x_ihs.shape
            for iter_round in range(num_iters):
                # Error performance
                x_iter = x_hist[:,iter_round][:,np.newaxis]
                df[sk_method,'Coefficient Error'][iter_round] += prediction_error(A_train,x_opt,x_iter)
                df[sk_method, 'Test Error'][iter_round] += test_mse(A_test,x_iter,y_test)
                # Timing performance
                df[sk_method,'Sketch'][iter_round] += ihs_total_time['Sketch'][iter_round]
                df[sk_method,'SVD'][iter_round]    += ihs_total_time['SVD'][iter_round]
                df[sk_method,'Solve'][iter_round]  += ihs_total_time['Solve'][iter_round]

            _total_time = ihs_total_time['Total']
            #print(f'IHS:{sk_method} Time: { _total_time:.4f}')
    df /= num_trials
    df.to_csv(path_or_buf=file_path,index=False)
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=6000, 
        help='''Number of rows in the dataset.''')
    parser.add_argument('-d', type=int, default=200, 
        help='''Number of trials to repeat the experiment.''')
    parser.add_argument('-i', type=int, default=1, 
        help='''Number of iterations for each IHS.''')
    parser.add_argument('-t', type=int, default=1, 
        help='''Number of trials to repeat the experiment.''')
    
    args = parser.parse_args()
    main(**vars(args))