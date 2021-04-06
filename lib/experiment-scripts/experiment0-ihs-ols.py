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

    Figure 1 https://jmlr.org/papers/volume17/14-460/14-460.pdf

    Evaluates the error to the model weights.
    """
    # * Experimental setup 
    file_path = 'results/experiment0-ihs-ols.csv'
    nn  = np.array([100*2**_ for _ in range(11)])
    d = 10
    num_trials = 10

    total_runs = len(nn)*num_trials*len(models)*len(methods) # For progress bar
    runs_complete = 0

    # * Results setup 
    results_df = pd.DataFrame()
    results_df['Rows'] = nn
    opt_results = np.zeros_like(nn,dtype=float)
    classical_results = np.zeros_like(opt_results)
    ihs_results = np.zeros_like(opt_results)
    classical_errors = {m : np.zeros_like(opt_results) for m in methods}
    ihs_errors = {m : np.zeros_like(opt_results) for m in methods}




    for i,n in enumerate(nn):
        for t in range(num_trials):
            # * 0. Data setup and sparsification
            y, A, x_model = experimental_data(n,d,1.0,seed=t*1000)
            sparse_data = SparseDataConverter(np.c_[A,y]).coo_data
            

            # * 1. Optimal weights use SVD instead of x_opt = np.linalg.lstsq(A,y,rcond=None)[0]
            x_opt = svd_solve(A,y)
            #x_opt = np.linalg.lstsq(A,y)[0]
            assert x_opt.shape == x_model.shape
            error =  prediction_error(A,x_model,x_opt) #  vec_error(x_model,x_opt) # 
            opt_results[i] += error

            #  * 2. Sketched weights
            #  2 Sketch parameters taken from IHS paper
            ihs_sk_dim = 7*d
            num_iters = 1 + int(np.ceil(np.log(n)))

            for sk_model, sk_method in itertools.product(models,methods):
                if runs_complete % 100 == 0:
                    print(f'Testing n={n}')
                    print(f'{runs_complete} of {total_runs} total runs completed.')
                if sk_model == 'Classical':
                    #  * 2a Classical
                    classical_sk_dim = num_iters*ihs_sk_dim # Scale up so same number of projections used.
                    classical_sk_solver = ClassicalSketch(n,d,classical_sk_dim,sk_method, sparse_data)
                    x_sk = classical_sk_solver.fit(A,y,seed=t)
                    assert x_opt.shape == x_sk.shape
                    classical_errors[sk_method][i] += prediction_error(A,x_model,x_sk)
                    runs_complete += 1
                elif sk_model == 'IHS':
                    # # * 2b IHS
                    ihs_solver = IterativeHessianOLS(n,d,ihs_sk_dim,sk_method)
                    x_ihs,_ = ihs_solver.fit(A,y)
                    assert x_opt.shape == x_ihs.shape
                    ihs_errors[sk_method][i] += prediction_error(A,x_model,x_ihs)
                    runs_complete += 1
            # #  * 2a Classical
            # classical_sk_dim = num_iters*ihs_sk_dim # Scale up so same number of projections used.
            # classical_sk_solver = ClassicalSketch(n,d,classical_sk_dim,'SJLT', sparse_data)
            # x_sk = classical_sk_solver.fit(A,y,seed=t)
            # assert x_opt.shape == x_sk.shape
            # classical_error =  prediction_error(A,x_model,x_sk) #  vec_error(x_model,x_sk) # 
            # classical_results[i] += classical_error
            
            # # # * 2b IHS
            # ihs_solver = IterativeHessianOLS(n,d,ihs_sk_dim,'SRHT')
            # x_ihs,_ = ihs_solver.fit(A,y)
            # assert x_opt.shape == x_ihs.shape
            # ihs_error =  prediction_error(A,x_model,x_ihs) #  vec_error(x_model,x_ihs) #  
            # ihs_results[i] += ihs_error
            
        opt_results[i] /= num_trials
    results_df['Optimal'] = opt_results
    for k,v in classical_errors.items():
        results_df['Classical ' + k] = classical_errors[k] / num_trials
    for k,v in ihs_errors.items():
        results_df['IHS ' + k] = ihs_errors[k] / num_trials
    print(results_df)
    results_df.to_csv(path_or_buf=file_path,index=False)
    
if __name__ == '__main__':
    main()