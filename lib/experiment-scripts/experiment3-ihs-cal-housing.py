import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.datasets import fetch_california_housing
import itertools
from scipy import sparse
from scipy.sparse import coo_matrix, csr_matrix
import os
import sys
sys.path.append('..')
from timeit import default_timer as timer
from sparse_data_converter import SparseDataConverter
from classical_sketch import ClassicalSketch
from iterative_hessian_sketch import IterativeHessianOLS
from experiment_utils import models, methods, experimental_data,svd_solve, prediction_error, test_mse,sparsify_data




def main(i,t,sparsify,s):
    """
    Task: IHS-OLS with random projections on real data, specifically, the CountSketch
    but with wall clock time to understand how the sketches fare in terms 
    of time.

    Evaluates the error as a function of the iterations with respect 
    to the optimal weights for the CALIFORNIA HOUSING DATASET.

    We use a default train/test split of 0.9 train and 0.1 test

    /home/dickens/code/thesis-experiments/datasets
    """
    # * Experimental setup      
    # *  2 Sketch parameters taken from IHS paper
    cwd = os.getcwd()
    data_name = 'cal_housing' 
    tra_data_path = '../../datasets/cal_housing_train.npy'
    val_data_path = '../../datasets/cal_housing_validate.npy'
    tes_data_path = '../../datasets/cal_housing_test.npy'
    training_data = np.load(tra_data_path)
    valid_data = np.load(val_data_path)
    test_data = np.load(tes_data_path)
    Xy = np.concatenate((training_data,valid_data),axis=0)
    X_train_raw, y_train = Xy[:,:-1], Xy[:,-1]
    X_test, y_test = test_data[:,:-1], test_data[:,-1]
    y_train = y_train[:,np.newaxis]
    y_test = y_test[:, np.newaxis]
    n_train,d = X_train_raw.shape

    # ! Parameter settings
    # sparsify = True
    # s = 0.125
    num_iters = i
    num_trials = t
    ihs_sketch_dims = [10*d]
    
    # * Results setup 
    file_path = 'results/experiment3-' + data_name 
    metrics = ['Sketch','SVD','Solve','Coefficient Error', 'Test Error']
    methods_and_exact = ['Exact(SVD)', 'Classical'] + methods
    all_setups = list(itertools.product(methods, ihs_sketch_dims)) 
    df = pd.DataFrame(np.zeros((num_iters,len(methods_and_exact)*len(metrics))))
    df.columns = pd.MultiIndex.from_product([methods_and_exact,metrics])
    ihs_results_opt = {mg: np.zeros(num_iters,dtype=float) for mg in all_setups}

    for t in range(num_trials):
        print(f'Trial {t}')
        seed = 500000*t
        np.random.seed(seed)
        #Xy = np.concatenate((training_data,valid_data),axis=0)
        np.random.shuffle(Xy)
        X_train_raw, y_train = Xy[:,:-1], Xy[:,-1]
        y_train = y_train[:,np.newaxis]
        if sparsify:
            rank_count = 1
            X_train = sparsify_data(X_train_raw,sparsity=s,seed=seed)
            
            while np.linalg.matrix_rank(X_train) != d:
                if rank_count < 10:
                    print('IN SPARSIFY')
                    X_train = sparsify_data(X_train_raw,sparsity=s,seed=seed+10*rank_count)
                    rank_count += 1
                else:
                    raise Exception(f'Data rank-deficient more than 10 times at sparsity {s}.\
                    Try decreasing number of trials or increasing sparsity.') 

        else:
            X_train = X_train_raw


        # * 0. Data setup and sparsification
        # A_train, A_test = A[train_ids,:], A[test_ids,:]
        # y_train, y_test = y[train_ids,:], y[test_ids,:]
        # A_train, A_test = A[train_ids,:].tocoo(), A[test_ids,:]
        # y_train, y_test = y[train_ids,:], y[test_ids,:]
        scaler =  Normalizer() # StandardScaler(with_mean=False) #  #
        A_train = scaler.fit_transform(X_train)
        A_test = scaler.transform(X_test)
        A_train_sparse = coo_matrix(A_train)
        

    #     # * densify the data if need be:
        # if sparse.issparse(A_train):
        #     A_train_dense = A_train.toarray()
        # else:
        #     A_train_dense = A_train
        #sparse_data = sparse.coo_matrix(A_train) 
        SVD_TIMER = timer()
        x_opt = svd_solve(A_train,y_train)
        svd_time = timer() - SVD_TIMER 
        #print(x_opt.shape, d)
        assert x_opt.shape == (d,1)
        df['Exact(SVD)','SVD'] += svd_time
        df['Exact(SVD)', 'Test Error'] += test_mse(A_test,x_opt,y_test)

    #     # ! Sparse methods using NUMBA need to compile the sketch so let's do that ahead of time
    #     # ! so that the timing experiment is not compromised.
        A_train_sparse = sparse.coo_matrix(A_train) 
        if t == 0:
            c_ihs_solver = IterativeHessianOLS(n_train,d,ihs_sketch_dims[0],'CountSketch',sparse_data=A_train_sparse)
            s_ihs_solver = IterativeHessianOLS(n_train,d,ihs_sketch_dims[0],'SJLT',sparse_data=A_train_sparse)
            ctsk_setup = c_ihs_solver.fit(A_train,y_train,num_iters,timing=True)
            sjlt_setup = s_ihs_solver.fit(A_train,y_train,num_iters,timing=True)

        # * 2b IHS
        for sk_method_gamma_d in all_setups:
            sk_method, ihs_sk_dim = sk_method_gamma_d[0], sk_method_gamma_d[1]
            ihs_solver = IterativeHessianOLS(n_train,d,ihs_sk_dim,sk_method,sparse_data=A_train_sparse)
            if sk_method == 'CountSketch' or sk_method == 'SJLT':
                x_ihs,x_hist,ihs_total_time = ihs_solver.fit(A_train_sparse,y_train,num_iters,timing=True)
            else:
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

        #  * 2a Classical
        classical_sk_dim = num_iters*ihs_sk_dim # Scale up so same number of projections used.
        classical_sk_solver = ClassicalSketch(n_train,d,classical_sk_dim,'CountSketch', sparse_data=A_train_sparse)
        x_sk,classical_times = classical_sk_solver.fit(A_train_sparse,y_train,seed=t,timing=True)
        assert x_opt.shape == x_sk.shape

        # Error performance
        df['Classical','Coefficient Error'] += prediction_error(A_train,x_opt,x_sk)
        df['Classical','Test Error'] += test_mse(A_test,x_sk,y_test)

        df['Classical','Sketch'] += classical_times['Sketch']
        df['Classical','SVD'] += classical_times['SVD']
        df['Classical', 'Solve'] += classical_times['Solve']

    # # #         _total_time = ihs_total_time['Total']
    # # #         #print(f'IHS:{sk_method} Time: { _total_time:.4f}')
    df /= num_trials
    print(df)
    if sparsify:
        file_path += str(s) + '.csv'
    else:
        file_path += '.csv'
    df.to_csv(path_or_buf=file_path ,index=False)
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('-n', type=int, default=10000, 
    #     help='''Number of rows in the dataset.''')
    # parser.add_argument('-d', type=bool, default=True, 
    #     help='''Bool: use all columns of the data.''')
    parser.add_argument('-i', type=int, default=1, 
        help='''Number of iterations for each IHS.''')
    parser.add_argument('-t', type=int, default=1, 
        help='''Number of trials to repeat the experiment.''')
    parser.add_argument('-sparsify', type=int, nargs='?', default=0, 
        help='''INT(Boolean) determining whether to sparsify the data.''')
    parser.add_argument('-s', type=float,nargs='?', default=0.1,
        help='''Sparsity parameter for the dataset.''')
    
    args = parser.parse_args()
    main(**vars(args))