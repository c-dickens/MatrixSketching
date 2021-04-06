import sys
sys.path.append('../')
import numpy as np
import pandas as pd
import itertools
from sparse_data_converter import SparseDataConverter
from scipy import sparse
from scipy.sparse import coo_matrix
from classical_sketch import ClassicalSketch
from iterative_hessian_sketch import IterativeHessianOLS
from experiment_utils import models, methods, experimental_data,svd_solve, prediction_error, vec_error




def main():
    """
    Task: IHS-OLS with random projections, specifically, the CountSketch

    Figure 2 https://jmlr.org/papers/volume17/14-460/14-460.pdf

    Evaluates the error as a function of the iterations with respect to the optimal 
    weights and the model weights.
    """
    # * Experimental setup      
    # *  2 Sketch parameters taken from IHS paper
    file_path = 'results/experiment0-ihs-ols.csv'
    n = 6000
    d = 200
    num_trials = 1
    num_iters = 10
    ihs_sketch_dims = [5*d, 10*d]
    all_setups = itertools.product(methods, ihs_sketch_dims)
    for i in all_setups:
        print(i)
    # * Results setup 
    opt_df = pd.DataFrame()
    #results_df['Rows'] = nn
    #opt_results = np.zeros_like(nn,dtype=float)
    # classical_results_opt = np.zeros(num_iters,dtype=float)
    # classical_results_model = np.zeros_like(classical_results_opt)
    ihs_results_opt = {mg: np.zeros(num_iters,dtype=float) for m in all_setups}
    ihs_results_model = {mg : np.zeros(num_iters,dtype=float) for m in all_setups}
    # # ihs_errors_opt = {m : np.zeros(num_iters,dtype=float) for m in methods}
    # # ihs_errors_model = {m : np.zeros(num_iters,dtype=float) for m in methods}

    for t in range(num_trials):
        print(f'Trial {t}')
         # * 0. Data setup and sparsification
        y, A, x_model = experimental_data(n,d,1.0,seed=1000)
        sparse_data = sparse.coo_matrix(A) #SparseDataConverter(A).coo_data
        x_opt = svd_solve(A,y)
        assert x_opt.shape == x_model.shape
        opt_model_error =  prediction_error(A,x_model,x_opt)
        
    #     # # * 1. Optimal weights use SVD instead of x_opt = np.linalg.lstsq(A,y,rcond=None)[0]
    #     # x_opt = svd_solve(A,y)
    #     # assert x_opt.shape == x_model.shape
    #     # error =  prediction_error(A,x_model,x_opt) #  vec_error(x_model,x_opt) # 
    #     #opt_results[i] += error

    #     #  * 2. Sketched weights
    
        
    #     # iterate over all of the different methods
    #     # for sk_method in methods:
            
    #     #     ihs_solver = IterativeHessianOLS(n,d,ihs_sk_dim,sk_method,sparse_data)
    #     #     x_ihs,x_hist = ihs_solver.fit(A,y,num_iters)
    #     #     assert x_opt.shape == x_ihs.shape
    #     #     for iter_round in range(num_iters):
    #     #         x_iter = x_hist[:,iter_round][:,np.newaxis]
    #     #         ihs_errors_opt[sk_method][iter_round] += prediction_error(A,x_opt,x_iter)
    #     #         ihs_errors_model[sk_method][iter_round] += prediction_error(A,x_model,x_iter)
    #     #     ihs_error_opt =  prediction_error(A,x_opt,x_ihs)
    #     #     ihs_error_model =  prediction_error(A,x_model,x_ihs) #  vec_error(x_model,x_ihs) #  
    #     #     ihs_results_opt[i] += prediction_error(A,x_opt,x_ihs)
    #     #     ihs_results_model[i] += ihs_error_model
            

    #     #  * 2a Classical
    #     # classical_sk_dim = num_iters*ihs_sk_dim # Scale up so same number of projections used.
    #     # classical_sk_solver = ClassicalSketch(n,d,classical_sk_dim,'SJLT', sparse_data)
    #     # x_sk = classical_sk_solver.fit(A,y,seed=t)
    #     # assert x_opt.shape == x_sk.shape
    #     # classical_error_opt =  prediction_error(A,x_opt,x_sk)
    #     # classical_error_model =  prediction_error(A,x_model,x_sk) #  vec_error(x_model,x_sk) # 
    #     # classical_results_opt += classical_error_opt
    #     # classical_results_model += classical_error_model
        
        # # * 2b IHS
        for sk_method_gamma_d in all_setups:
            sk_method, ihs_sk_dim = sk_method_gamma_d[0], sk_method_gamma_d[1]
            ihs_solver = IterativeHessianOLS(n,d,ihs_sk_dim,sk_method,sparse_data=sparse_data)
            x_ihs,x_hist = ihs_solver.fit(A,y,num_iters)
            assert x_opt.shape == x_ihs.shape
            for iter_round in range(num_iters):
                x_iter = x_hist[:,iter_round][:,np.newaxis]
                ihs_results_opt[sk_method_gamma_d][iter_round] += prediction_error(A,x_opt,x_iter)
                ihs_results_model[sk_method_gamma_d][iter_round] += prediction_error(A,x_model,x_iter)
                print(ihs_results_opt[sk_method_gamma_d][iter_round])
            # ihs_error_opt =  prediction_error(A,x_opt,x_ihs)
            # ihs_results_opt += prediction_error(A,x_opt,x_ihs)
            # ihs_results_model += prediction_error(A,x_model,x_ihs)

    # # # for k,v in ihs_errors_opt.items():
    # # #     opt_df[k] = v / num_trials
    # # for result_arr in [classical_results_opt, classical_results_model,ihs_results_opt, ihs_results_model]:
    # #     result_arr[i] /= num_trials 
    # # opt_df['Classical'] = classical_results_opt
    for sk_method_gamma_d in all_setups:
        sk_method, ihs_sk_dim = sk_method_gamma_d[0], sk_method_gamma_d[1]
        gamma = int(d/ihs_sk_dim)
        column_name = sk_method + str(gamma)
        print('Sketch method: ',sk_method)
        opt_df[column_name] = ihs_results_opt[sk_method_gamma_d] / num_trials
    print(opt_df[:num_iters])
    #results_df.to_csv(path_or_buf=file_path,index=False)
    
if __name__ == '__main__':
    main()