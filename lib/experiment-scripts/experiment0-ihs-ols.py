import sys
sys.path.append('../')
import numpy as np
import pandas as pd
from sparse_data_converter import SparseDataConverter
from scipy import sparse
from scipy.sparse import coo_matrix
from classical_sketch import ClassicalSketch
from experiment_utils import svd_solve, prediction_error

def experimental_data(n,d, sigma=1.0,seed=100):
    """
    The data for this experiment is taken from 3.1 https://jmlr.org/papers/volume17/14-460/14-460.pdf
    A       : n \times d matrix of N(0,1) entries
    x_model : d \times 1 vector chosen uniformly from the d dimensional sphere by choosing 
              random normals and then normalising.
    w       : Vector of noise perturbation which is N(0,sigma**2*I_n)
    y = A@x_model + w
    """
    np.random.seed(seed)
    A = np.random.randn(n,d)
    x_model = np.random.randn(d,1)
    x_model /= np.linalg.norm(x_model,axis=0)
    w = sigma**2*np.random.randn(n,1)
    y = A@x_model + w
    return y,A,x_model


def main():
    """
    Task: IHS-OLS with random projections, specifically, the CountSketch

    Figure 1 https://jmlr.org/papers/volume17/14-460/14-460.pdf
    """
    # * Experimental setup 
    nn  = np.array([100*2**_ for _ in range(5)])
    d = 10
    num_trials = 5

    # * Results setup 
    results_df = pd.DataFrame()
    results_df['Rows'] = nn
    opt_results = np.zeros_like(nn,dtype=float)
    classical_results = np.zeros_like(opt_results)

    for i,n in enumerate(nn):
        for t in range(num_trials):
            # * 0. Data setup and sparsification
            y, A, x_model = experimental_data(n,d,seed=t)
            sparse_data = SparseDataConverter(np.c_[A,y]).coo_data
            

            # * 1. Optimal weights use SVD instead of x_opt = np.linalg.lstsq(A,y,rcond=None)[0]
            x_opt = svd_solve(A,y)
            assert x_opt.shape == x_model.shape
            error =  prediction_error(A,x_model,x_opt) # vec_error(x_model,x_opt) #
            opt_results[i] += error

            #  * 2. Sketched weights
            #  2 Sketch parameters taken from IHS paper

            #  * 2a Classical
            classical_sk_dim = 1 + int(np.ceil(np.log(n)))
            classical_sk_dim *= 7*d
            classical_sk_solver = ClassicalSketch(n,d,classical_sk_dim,'SJLT', sparse_data)
            x_sk = classical_sk_solver.fit(A,y,seed=t)
            assert x_opt.shape == x_sk.shape
            classical_error =  prediction_error(A,x_model,x_sk) # vec_error(x_model,x_sk) #
            classical_results[i] += classical_error
            
            # # * 2b IHS


        opt_results[i] /= num_trials
        classical_results[i] /= num_trials
    results_df['Optimal'] = opt_results
    results_df['Classical'] = classical_results
    print(results_df)
    
if __name__ == '__main__':
    main()