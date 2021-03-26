import numpy as np
import pandas as pd
from experiment_utils import prediction_error

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
    num_trials = 2

    # * Results setup 
    results_df = pd.DataFrame()
    results_df['Rows'] = nn
    opt_results = np.zeros_like(nn,dtype=float)

    for i,n in enumerate(nn):
        for t in range(num_trials):
            y, A, x_model = experimental_data(n,d,seed=t)
            x_opt = np.linalg.lstsq(A,y)[0]
            error = prediction_error(A,x_model,x_opt)
            opt_results[i] += error
        opt_results[i] /= num_trials
    results_df['Optimal'] = opt_results
    print(results_df)
    print(opt_results)
if __name__ == '__main__':
    main()