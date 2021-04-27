import sys
sys.path.append('../')
sys.path.append('../ihs-experiment-scripts')
from experiment_utils import shi_phillips_ridge_data, svd_ridge_solve, get_euclidean_errors
from iterative_ridge_regression import IterativeRidge
from sklearn.preprocessing import StandardScaler
import numpy as np
from math import floor, exp
from scipy.fftpack import dct
import json
import matplotlib.pyplot as plt

def error_vs_sketch_size(n,d,sketch_size,r):
    # * Experimental parameters
    trials = 5
    iterations = 5
    eff_rank = int(floor(r*d + 0.5))
    g_range = [-1,0,1,2]
    
    # * Results data structures
    fd_errors = {g : np.zeros((iterations+1, trials),dtype=float) for g in g_range}
    rfd_errors = {g : np.zeros((iterations+1, trials),dtype=float) for g in g_range}
    for t in range(trials):
        print('*'*10, '\t TRIAL ', t, '\t R = ', r, '\t', '*'*10)
        X,y,w0 = shi_phillips_ridge_data(n,d,eff_rank,tail_strength=0.125,seed=t)

        X = StandardScaler().fit_transform(X)
        gammas = [(2**a)*(np.linalg.norm(X,ord='fro')**2)/sketch_size for a in g_range]
        for i, g in enumerate(gammas):
            print('#'*10, f'\t GAMMA: G={g}, i={i} a={g_range[i]}\t', '#'*10)
            a = g_range[i]

            # ! Optimal solution
            x_opt = svd_ridge_solve(X,y,g)

            # ! FD Sketching 
            fdr = IterativeRidge(n,d,sk_dim=sketch_size,sk_mode='FD',gamma=g)
            x_fd, all_fd_x,fd_measured = fdr.fast_iterate(X,y,iterations)
            fd_errors[a][:,t] =  get_euclidean_errors(all_fd_x,x_opt)

            
            # ! RFD Sketching  #  (fd_dim=sketch_size,fd_mode='RFD',gamma=g)
            # rfdr = IterativeRidge(fd_dim=sketch_size,gamma=g,fd_mode='RFD')#(n,d,sketch_size, sk_mode='RFD')
            rfdr = IterativeRidge(n,d,sk_dim=sketch_size,sk_mode='RFD',gamma=g)
            x_rfd, rfd_all_x, rfd_measured = rfdr.fast_iterate(X,y,iterations)
            rfd_errors[a][:,t] = get_euclidean_errors(rfd_all_x,x_opt)
    print(fd_errors)
    print(rfd_errors)
    # ! Results setup  and plotting
    fig, ax = plt.subplots()
    for i,g in enumerate(gammas):
        a = g_range[i]

        # * FD
        fd_errs = fd_errors[a]
        fd_mean_errs = np.mean(fd_errs,axis=1)
        fd_std_errs = np.std(fd_errs,axis=1)
        ax.errorbar(range(iterations+1),fd_mean_errs,yerr=fd_std_errs, label=f'g=2^{a}*(1/m)')

        # * RFD
        rfd_errs = rfd_errors[a]
        rfd_mean_errs = np.mean(rfd_errs,axis=1)
        rfd_std_errs = np.std(rfd_errs,axis=1)
        ax.errorbar(range(iterations+1),rfd_mean_errs,yerr=rfd_std_errs, label=f'Rg=2^{a}*(1/m)')
    ax.legend()
    ax.set_yscale('log',base=10)
    ax.legend()
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Error')
    path = 'results/synthetic' 
    fname = path + '/bound-test-' + str(r) + '.png'
    fig.savefig(fname,dpi=200,bbox_inches='tight',pad_inches=None)

    # ! Prepare and save the results in json format 
    res_name = 'results/synthetic/bound-test' + str(r) + '.json'
    for d in [fd_errors,rfd_errors]:
        for k,v in d.items():
            if type(v) == np.ndarray:
                d[k] = v.tolist()
    results = {
        'FD Error'   : fd_errors,
        'RFD Error'   : rfd_errors,
        }
    with open(res_name, 'w') as fp:
        json.dump(results, fp,sort_keys=True, indent=4)


   
def main():
    n = 2**15
    d = 2**12
    for (m,r) in [(300,0.1),(300,0.5)]: 
        error_vs_sketch_size(n,d,m,r) 

if __name__ =='__main__':
    main()
