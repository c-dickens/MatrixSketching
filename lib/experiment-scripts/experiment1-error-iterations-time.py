import sys
sys.path.append('../')
import numpy as np
import pandas as pd
from sparse_data_converter import SparseDataConverter
from scipy import sparse
from scipy.sparse import coo_matrix
from classical_sketch import ClassicalSketch
from iterative_hessian_sketch import IterativeHessianOLS
from experiment_utils import experimental_data,svd_solve, prediction_error, vec_error




def main():
    """
    Task: IHS-OLS with random projections, specifically, the CountSketch

    Figure 2 https://jmlr.org/papers/volume17/14-460/14-460.pdf

    Evaluates the error as a function of the iterations with respect to the optimal 
    weights and the model weights.
    """
    # * Experimental setup 
    file_path = 'results/experiment0-ihs-ols.csv'
    nn  = [6000]
    d = 200
    num_trials = 1
    num_iters = 10

    # * Results setup 
    opt_df = pd.DataFrame()
    #results_df['Rows'] = nn
    #opt_results = np.zeros_like(nn,dtype=float)
    classical_results_opt = np.zeros(num_iters,dtype=float)
    classical_results_model = np.zeros_like(classical_results_opt)
    ihs_results_opt = np.zeros_like(classical_results_opt)
    ihs_results_model = np.zeros_like(classical_results_opt)

    for i,n in enumerate(nn):
        for t in range(num_trials):
            # * 0. Data setup and sparsification
            y, A, x_model = experimental_data(n,d,1.0,seed=t)
            sparse_data = SparseDataConverter(np.c_[A,y]).coo_data
            

            # * 1. Optimal weights use SVD instead of x_opt = np.linalg.lstsq(A,y,rcond=None)[0]
            x_opt = svd_solve(A,y)
            assert x_opt.shape == x_model.shape
            error =  prediction_error(A,x_model,x_opt) #  vec_error(x_model,x_opt) # 
            #opt_results[i] += error

            #  * 2. Sketched weights
            #  2 Sketch parameters taken from IHS paper
            ihs_sk_dim = 10*d
            

            #  * 2a Classical
            classical_sk_dim = num_iters*ihs_sk_dim # Scale up so same number of projections used.
            classical_sk_solver = ClassicalSketch(n,d,classical_sk_dim,'SJLT', sparse_data)
            x_sk = classical_sk_solver.fit(A,y,seed=t)
            assert x_opt.shape == x_sk.shape
            classical_error_opt =  prediction_error(A,x_opt,x_sk)
            classical_error_model =  prediction_error(A,x_model,x_sk) #  vec_error(x_model,x_sk) # 
            classical_results_opt += classical_error_opt
            classical_results_model += classical_error_model
            
            # # * 2b IHS
            ihs_solver = IterativeHessianOLS(n,d,ihs_sk_dim,'SRHT')
            x_ihs,x_hist = ihs_solver.fit(A,y,num_iters)
            assert x_opt.shape == x_ihs.shape
            for iter_round in range(num_iters):
                x_iter = x_hist[:,iter_round][:,np.newaxis]
                ihs_results_opt[iter_round] += prediction_error(A,x_opt,x_iter)
                ihs_results_model[iter_round] += prediction_error(A,x_model,x_iter)
            # ihs_error_opt =  prediction_error(A,x_opt,x_ihs)
            # ihs_error_model =  prediction_error(A,x_model,x_ihs) #  vec_error(x_model,x_ihs) #  
            # ihs_results_opt[i] += prediction_error(A,x_opt,x_ihs)
            # ihs_results_model[i] += ihs_error_model


    for result_arr in [classical_results_opt, classical_results_model,ihs_results_opt, ihs_results_model]:
        result_arr[i] /= num_trials 
        # classical_results[i] /= num_trials
        # ihs_results[i] /= num_trials
    opt_df['Classical'] = classical_results_opt
    opt_df['IHS'] = ihs_results_opt
    print(opt_df)
    #results_df.to_csv(path_or_buf=file_path,index=False)
    
if __name__ == '__main__':
    main()