import matplotlib.pyplot as plt
import json
import numpy as np
import pprint
import sys
sys.path.append('/home/dickens/code/thesis-experiments/lib/ihs-experiment-scripts/results')
from plot_config import fd_synthetic_params,rfd_synthetic_params, fd_experiment0_markers
import tikzplotlib

OUT_DIR = 'synthetic/'

def experiment0_synthetic_plots():
    """
    Plots the error vs iterations profile
    """
    fig, axes = plt.subplots(dpi=150,ncols=2)

    for j,f in enumerate(['bound-test0.1.json', 'bound-test0.5.json']):
        print(f)
        with open(f) as json_file:
            data = json.load(json_file)
        gammas = list(data['FD Error'].keys())
        g_range = gammas
        fd_errors = data['FD Error']
        rfd_errors = data['RFD Error']
        iterations = 5
        ax = axes[j]

    #     ax.plot([], [],label='FD',color='white')
    #     ax.plot([], [],label='RFD',color='white')
        for i,g in enumerate(gammas):
            a = g_range[i]

            # * FD
            fd_errs = fd_errors[a]
            fd_mean_errs = np.mean(fd_errs,axis=1)
            fd_std_errs = np.std(fd_errs,axis=1)
            ax.plot(range(iterations+1),fd_mean_errs,
                        label=f'$Fa={a}$',**fd_synthetic_params,**fd_experiment0_markers[a])

            # * RFD
            rfd_errs = rfd_errors[a]
            rfd_mean_errs = np.mean(rfd_errs,axis=1)
            rfd_std_errs = np.std(rfd_errs,axis=1)
            ax.plot(range(iterations+1),rfd_mean_errs,
                    label=f'$Ra={a}$',**rfd_synthetic_params,**fd_experiment0_markers[a])
        ax.set_yscale('log',base=10)
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Error')
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [0,2,4,6,8,1,3,5,7,9]
    # plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
    # leg = ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order],
    #                 loc='upper center', bbox_to_anchor=(0., 1.25),
    #                 title=r'$\gamma = 2^a |A|_F^2 / m$', ncol=4,frameon=False,)
    leg = ax.legend(loc='upper center', bbox_to_anchor=(0., 1.25),
                    title=r'$\gamma = 2^a |A|_F^2 / m$', ncol=4,frameon=False,)
    leg._legend_box.align = "left"
    out_fname = OUT_DIR + '-ifdrr-error-iterations.tex'
    tikzplotlib.save(out_fname)
    # with open(fname) as json_file:
    #     data = json.load(json_file)

    # gammas = list(data['FD Error'].keys())
    # g_range = gammas
    # fd_errors = data['FD Error']
    # rfd_errors = data['RFD Error']
    # iterations = 5

    # fig, ax = plt.subplots(dpi=150)
    # ax.plot([], [],label='FD',color='white')
    # ax.plot([], [],label='RFD',color='white')
    # for i,g in enumerate(gammas):
    #     a = g_range[i]

    #     # * FD
    #     fd_errs = fd_errors[a]
    #     fd_mean_errs = np.mean(fd_errs,axis=1)
    #     fd_std_errs = np.std(fd_errs,axis=1)
    #     ax.plot(range(iterations+1),fd_mean_errs,
    #                 label=f'$Fa={a}$',**fd_synthetic_params,**fd_experiment0_markers[a])

    #     # * RFD
    #     rfd_errs = rfd_errors[a]
    #     rfd_mean_errs = np.mean(rfd_errs,axis=1)
    #     rfd_std_errs = np.std(rfd_errs,axis=1)
    #     ax.plot(range(iterations+1),rfd_mean_errs,
    #             label=f'$Ra={a}$',**rfd_synthetic_params,**fd_experiment0_markers[a])
        
    # handles, labels = plt.gca().get_legend_handles_labels()
    # order = [0,2,4,6,8,1,3,5,7,9]
    # leg = ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order],
    #                 title=r'$\gamma = 2^a |A|_F^2 / m$', ncol=2,loc='lower left',frameon=False,)
    # leg._legend_box.align = "left"
    # ax.set_yscale('log',base=10)
    # ax.set_xlabel('Iterations')
    # ax.set_ylabel('Error')
    # out_fname = OUT_DIR + fname[:-5] + '-ifdrr-error-iterations.tex'
    # tikzplotlib.save(out_fname)

def main():
    experiment0_synthetic_plots()

