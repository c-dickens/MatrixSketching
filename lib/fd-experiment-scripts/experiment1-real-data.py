#from data_factory import DataFactory
# from frequent_directions_aistats_ridge_regression import FDRidge
# from random_projections_aistats_ridge_regression import RPRidge
import sys
sys.path.append('..')
sys.path.append('../ihs-experiment-scripts')
from sklearn.linear_model import Ridge,LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.kernel_approximation import RBFSampler
#from plot_config import fd_params, rfd_params, sjlt_rp_params, gauss_rp_params, gauss_hs_params, sjlt_hs_params 
import numpy as np
from math import floor
import json
import matplotlib.pyplot as plt
#from utils import get_errors, get_x_opt
from timeit import default_timer as timer
import pprint
from iterative_ridge_regression import IterativeRidge
from experiment_utils import svd_ridge_solve, get_euclidean_errors
from scipy.sparse import coo_matrix

def real_data_error_profile(data_name,sketch_size):
    '''
    Use polynomial feature map which generates {feature_size + degree \choose degree} new 
    features
    '''
    # * Experimental parameters
    n = 10000
    trials = 1
    iterations = 10
    # ds = DataFactory(n=n)
    if data_name == 'CoverType':
        _ = np.load('../../datasets/covertype.npy')
        _X,_y = _[:,:-1], _[:,-1]
        feature_expansion = 'Polynomial'
        features = [2]
    elif data_name == 'w8a':
        _ = np.load('../../datasets/w8a.npy')
        _X,_y = _[:,:-1], _[:,-1]
        feature_expansion = 'RBF'
        features = [2500]

    # * Results data structures
    exact_results = {
        'solve time' : { _ : np.zeros(trials,dtype=float) for _ in features}
    }
    fd_results =  { 
        'errors'          : { _ : np.zeros((iterations+1, trials),dtype=float) for _ in features},
        'build times'     : { _ : np.zeros(trials,dtype=float) for _ in features},
        'iteration times' : { _ : np.zeros(trials,dtype=float) for _ in features},
        'all_times'       : { _ : np.zeros((iterations+1, trials),dtype=float) for _ in features}
    }
    rfd_results = { 
        'errors'          : { _ : np.zeros((iterations+1, trials),dtype=float) for _ in features},
        'build times'     : { _ : np.zeros(trials,dtype=float) for _ in features},
        'iteration times' : { _ : np.zeros(trials,dtype=float) for _ in features},
        'all_times'       : { _ : np.zeros((iterations+1, trials),dtype=float) for _ in features}
    }
    rp_srht_results = { 
        'errors'          : { _ : np.zeros((iterations+1, trials),dtype=float) for _ in features},
        'build times'     : { _ : np.zeros(trials,dtype=float) for _ in features},
        'iteration times' : { _ : np.zeros(trials,dtype=float) for _ in features},
        'all_times'       : { _ : np.zeros((iterations+1, trials),dtype=float) for _ in features}
    }
    rp_cntsk_results = { 
        'errors'          : { _ : np.zeros((iterations+1, trials),dtype=float) for _ in features},
        'build times'     : { _ : np.zeros(trials,dtype=float) for _ in features},
        'iteration times' : { _ : np.zeros(trials,dtype=float) for _ in features},
        'all_times'       : { _ : np.zeros((iterations+1, trials),dtype=float) for _ in features}
    }
    ihs_srht_results =  { 
        'errors'          : { _ : np.zeros((iterations+1, trials),dtype=float) for _ in features},
        'build times'     : { _ : np.zeros(trials,dtype=float) for _ in features},
        'iteration times' : { _ : np.zeros(trials,dtype=float) for _ in features},
        'all_times'       : { _ : np.zeros((iterations+1, trials),dtype=float) for _ in features}
    }
    ihs_countsketch_results =  { 
        'errors'          : { _ : np.zeros((iterations+1, trials),dtype=float) for _ in features},
        'build times'     : { _ : np.zeros(trials,dtype=float) for _ in features},
        'iteration times' : { _ : np.zeros(trials,dtype=float) for _ in features},
        'all_times'       : { _ : np.zeros((iterations+1, trials),dtype=float) for _ in features}
    }


    mean_iter_time_single = lambda a : np.mean(a['all_times'][1:] - a['sketch time'])
    mean_iter_time_multi = lambda a : np.mean(a['all_times'][1:] - a['sketch time']/iterations)

    for t in range(trials):
        print('*'*10, '\t TRIAL ', t, '\t', '*'*10)
        np.random.seed(t)
        sample = np.random.choice(_X.shape[0],size=n, replace=False)
        X_sample, y = _X[sample], _y[sample]
        
        for i, feature_hyper in enumerate(features):
            
            print('######### FEATURIZING #########')
            if feature_expansion == 'Polynomial':
                X_poly = PolynomialFeatures(degree=feature_hyper).fit_transform(X_sample)
                if X_poly.shape[1] > X_poly.shape[0]:
                    nkeep = int(1.5*(X_poly.shape[0] - X_poly.shape[1]))
                    X_poly = X_poly[:,:nkeep]
            else:
                X_poly = RBFSampler(gamma=0.0001, random_state=t,n_components=feature_hyper).fit_transform(X_sample)
            X = StandardScaler().fit_transform(X_poly)
            N,D = X.shape
            X_train_sparse = coo_matrix(X) 
            g = np.linalg.norm(X,ord='fro')**2 / sketch_size 
            print('#'*10, f'\t GAMMA: G={g}, i={i} d={feature_hyper}\t', '#'*10)

            # # ! Optimal solution
            print('#'*60)
            print('Solving exactly: Data shape: ', X.shape)
            solve_start = timer()
            x_opt = svd_ridge_solve(X,y,g)
            solve_time = timer() - solve_start
            exact_results['solve time'][feature_hyper][t] = solve_time


            # ! FD Sketching 
            print('#'*10, '\t FREQUENT DIRECTIONS \t', '#'*10)
            #fdr = FDRidge(fd_dim=sketch_size,gamma=g)
            fdr = IterativeRidge(N,D,sk_dim=sketch_size,sk_mode='FD',gamma=g)
            _, all_x,fd_measured = fdr.fit(X,y,iterations)

            fd_results['errors'][feature_hyper][:,t]             =  get_euclidean_errors(all_x,x_opt)
            fd_results['build times'][feature_hyper][t]          =  fd_measured['sketch time']
            fd_results['iteration times'][feature_hyper][t]      =  mean_iter_time_single(fd_measured)
            fd_results['all_times'][feature_hyper][:,t]          =  fd_measured['all_times']


            # # ! RFD Sketching 
            print('#'*10, '\t ROBUST FREQUENT DIRECTIONS \t', '#'*10)
            #rfdr = FDRidge(fd_dim=sketch_size,fd_mode='RFD',gamma=g)
            rfdr = IterativeRidge(N,D,sk_dim=sketch_size,sk_mode='RFD',gamma=g)
            _, rfd_all_x,rfd_measured = rfdr.fit(X,y,iterations)
            
            rfd_results['errors'][feature_hyper][:,t]             =  get_euclidean_errors(rfd_all_x,x_opt)
            rfd_results['build times'][feature_hyper][t]          =  rfd_measured['sketch time']
            rfd_results['iteration times'][feature_hyper][t]      =  mean_iter_time_single(rfd_measured)
            rfd_results['all_times'][feature_hyper][:,t]          =  rfd_measured['all_times']

            # # ! Single Random sketches
            print('#'*10, '\t SRHT SINGLE \t', '#'*10)
            srht_single = IterativeRidge(N,D,sk_dim=sketch_size,sk_mode='SRHT',gamma=g) # RPRidge(rp_dim=sketch_size,rp_mode='Gaussian',gamma=g)
            _, srht_single_all_x, srht_single_measured = srht_single.fit(X,y,seed=i)

            rp_srht_results['errors'][feature_hyper][:,t]             =  get_euclidean_errors(srht_single_all_x,x_opt)
            rp_srht_results['build times'][feature_hyper][t]          =  srht_single_measured['sketch time']
            rp_srht_results['iteration times'][feature_hyper][t]      =  mean_iter_time_single(srht_single_measured)
            rp_srht_results['all_times'][feature_hyper][:,t]          =  srht_single_measured['all_times']

            print('#'*10, '\t CountSketch SINGLE \t', '#'*10)
            cntsk_single = IterativeRidge(N,D,sk_dim=sketch_size,sk_mode='CountSketch',gamma=g,sparse_data=X_train_sparse) #RPRidge(rp_dim=sketch_size,rp_mode='SJLT',gamma=g)
            _, cntsk_single_all_x, cntsk_single_measured = cntsk_single.fit(X,y)

            rp_cntsk_results['errors'][feature_hyper][:,t]             =  get_euclidean_errors(cntsk_single_all_x,x_opt)
            rp_cntsk_results['build times'][feature_hyper][t]          =  cntsk_single_measured['sketch time']
            rp_cntsk_results['iteration times'][feature_hyper][t]      =  mean_iter_time_single(cntsk_single_measured)
            rp_cntsk_results['all_times'][feature_hyper][:,t]          =  cntsk_single_measured['all_times']


            # ! Multi Random sketches           
            print('#'*10, '\t CountSketch IHS \t', '#'*10)
            ihs_cntsk = IterativeRidge(N,D,sk_dim=sketch_size,sk_mode='CountSketch',gamma=g,sparse_data=X_train_sparse,ihs_mode='multi') #RPRidge(rp_dim=sketch_size,rp_mode='SJLT',gamma=g)
            _, ihs_cntsk_all_x, ihs_cntsk_measured = ihs_cntsk.fit(X,y)
            
              
            ihs_countsketch_results['errors'][feature_hyper][:,t]              =  get_euclidean_errors(ihs_cntsk_all_x,x_opt)
            ihs_countsketch_results['build times'][feature_hyper][t]           =  np.sum(ihs_cntsk_measured['sketch time'])
            ihs_countsketch_results['iteration times'][feature_hyper][t]       =  mean_iter_time_multi(ihs_cntsk_measured)
            ihs_countsketch_results['all_times'][feature_hyper][:,t]           =  ihs_cntsk_measured['all_times']


            print('#'*10, '\t SRHT IHS \t', '#'*10)
            ihs_srht = IterativeRidge(N,D,sk_dim=sketch_size,sk_mode='SRHT',gamma=g,ihs_mode='multi')#  RPRidge(rp_dim=sketch_size,rp_mode='Gaussian',gamma=g)
            _, ihs_srht_all_x, ihs_srht_measured = ihs_srht.fit(X,y)

            ihs_srht_results['errors'][feature_hyper][:,t]              =  get_euclidean_errors(ihs_srht_all_x,x_opt)
            ihs_srht_results['build times'][feature_hyper][t]           =  np.sum(ihs_srht_measured['sketch time'])
            ihs_srht_results['iteration times'][feature_hyper][t]       =  mean_iter_time_multi(ihs_srht_measured)
            ihs_srht_results['all_times'][feature_hyper][:,t]           =  ihs_srht_measured['all_times']


    # ! Prepare and save the results in json format 
    pp = pprint.PrettyPrinter(indent=4)
    # print('FD')
    # pp.pprint(fd_results['errors'])
    # print('SRHT-Single')
    # pp.pprint(rp_srht_results['errors'])
    # print('SRHT-Multi')
    # pp.pprint(ihs_g_results['errors'])
    # # print('Gauss')
    # # pp.pprint(rpg_results)
    # print('ihs:SRHT')
    # pp.pprint(ihs_srht_results)
    # print('SJLT')
    # pp.pprint(rp_cntsk_results)
    # print('ihs:SJLT')
    # pp.pprint(ihs_countsketch_results)




    
    
    
    results_file_name = 'results/real_data/error_profile' + data_name + '.json'
    for d in [exact_results, fd_results, rfd_results, rp_srht_results, rp_cntsk_results, ihs_srht_results, ihs_countsketch_results]:
        for k,v in d.items():
            for v_key, v_val in v.items():
                if type(v_val) == np.ndarray:
                    d[k][v_key] = v_val.tolist()
    all_results = {
        'Exact'     : exact_results,
        'FD'        : fd_results,
        'RFD'       : rfd_results,
        'SRHT'      : rp_srht_results,
        'CountSketch'      : rp_cntsk_results,
        'ihs:SRHT' : ihs_srht_results,
        'ihs:CountSketch'  : ihs_countsketch_results
    }


    with open(results_file_name, 'w') as fp:
        json.dump(all_results, fp,sort_keys=True, indent=4)  


def main():
    datasets = ['CoverType']
    for d in datasets:
        real_data_error_profile(d,300) 


if __name__ =='__main__':
    main()