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
    file_path_opt = 'results/experiment1-ihs-iterations-opt.csv'
    file_path_model = 'results/experiment1-ihs-iterations-model.csv'
    n = 6000
    d = 200
    num_trials = 1
    num_iters = 10
    ihs_sketch_dims = [5*d, 10*d]
    
    # * Results setup 
    all_setups = list(itertools.product(methods, ihs_sketch_dims))
    opt_df = pd.DataFrame()
    model_df = pd.DataFrame()
    ihs_results_opt = {mg: np.zeros(num_iters,dtype=float) for mg in all_setups}
    ihs_results_model = {mg : np.zeros(num_iters,dtype=float) for mg in all_setups}

    for t in range(num_trials):
        print(f'Trial {t}')
         # * 0. Data setup and sparsification
        y, A, x_model = experimental_data(n,d,1.0,seed=1000)
        sparse_data = sparse.coo_matrix(A) #SparseDataConverter(A).coo_data
        x_opt = svd_solve(A,y)
        assert x_opt.shape == x_model.shape
        opt_model_error =  prediction_error(A,x_model,x_opt)

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
         
    # # opt_df['Classical'] = classical_results_opt
    for sk_method_gamma_d in all_setups:
        sk_method, ihs_sk_dim = sk_method_gamma_d[0], sk_method_gamma_d[1]
        gamma = int(ihs_sk_dim/d)
        column_name = sk_method + str(gamma)
        print('Sketch method: ',sk_method)
        opt_df[column_name] = ihs_results_opt[sk_method_gamma_d] / num_trials
        model_df[column_name] = ihs_results_model[sk_method_gamma_d] / num_trials
    print(opt_df[:num_iters])
    opt_df.to_csv(path_or_buf=file_path_opt,index=False)
    model_df.to_csv(path_or_buf=file_path_model,index=False)
    
if __name__ == '__main__':
    main()