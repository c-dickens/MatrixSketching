import matplotlib.pyplot as plt
import json
import numpy as np
import pprint
import sys
sys.path.append('/home/dickens/code/thesis-experiments/lib/ihs-experiment-scripts/results')
from plot_config import fd_real_data_plot_config
import tikzplotlib
import pandas as pd
import tikzplotlib


OUT_DIR = 'real_data/'
FPATH = 'real_data/error_profile-'


def build_time_plot(dataset,feature_hyper):
    """
    Plots the build times
    """
    file = FPATH + dataset + '.json'
    sparse_methods = ['CountSketch','SRHT', ]
    dense_methods = [ 'FD', 'RFD','ihs:CountSketch','ihs:SJLT', 'ihs:SRHT',]
    with open(file) as json_file:
        data_dict = json.load(json_file)
        
    dense_build_time_dict = {}
    sparse_build_time_dict = {}
    fig, axes = plt.subplots(dpi=150,ncols=2)
    ax_sparse, ax_dense = axes
    for k,v in data_dict.items():
        if k == 'Exact':
             continue
        # build_time_res = np.array(data_dict[k]['build times'][feature_hyper])
        # print(build_time_res.shape)
        build_time = np.array(data_dict[k]['build times'][feature_hyper])
        #print(k,build_time.shape)
        if build_time.ndim > 1:
            #print(build_time.shape)
            build_time = np.sum(build_time,axis=0)
        #print(build_time)
        build_time = np.mean(build_time)
        #print(build_time)
        
        if k in sparse_methods:
            sparse_build_time_dict[k] = build_time
        if k in dense_methods:
            dense_build_time_dict[k] = build_time

            
    sparse_build_time_df = pd.DataFrame.from_dict(sparse_build_time_dict,orient='index')
    dense_build_time_df = pd.DataFrame.from_dict(dense_build_time_dict,orient='index')
    sparse_build_time_df.plot.bar(ax=ax_sparse,rot=45,legend=False,)
    dense_build_time_df.plot.bar(ax=ax_dense,rot=45,legend=False)

    fpath = FPATH[:10] +  dataset + '_build_time'
    fig.savefig(fpath)
    out_fname = OUT_DIR +  dataset + '_build_time.tex'
    tikzplotlib.save(out_fname)

def update_time_plot(dataset,feature_hyper):
    """
    Plots the build times
    """
    file = FPATH + dataset + '.json'
    sparse_methods = ['CountSketch','FD', 'RFD', 'SRHT']
    dense_methods = [ 'ihs:CountSketch', 'ihs:SJLT', 'ihs:SRHT']
    with open(file) as json_file:
        data_dict = json.load(json_file)
        
    dense_build_time_dict = {}
    sparse_build_time_dict = {}
    fig, axes = plt.subplots(dpi=150,ncols=2)
    ax_sparse, ax_dense = axes
    for k,v in data_dict.items():
        if k == 'Exact':
             continue
        build_time = np.array(data_dict[k]['iteration times'][feature_hyper])
        if build_time.size > 1:
            build_time = np.sum(build_time)
        
        if k in sparse_methods:
            sparse_build_time_dict[k] = build_time
        if k in dense_methods:
            dense_build_time_dict[k] = build_time

            
    sparse_build_time_df = pd.DataFrame.from_dict(sparse_build_time_dict,orient='index')
    dense_build_time_df = pd.DataFrame.from_dict(dense_build_time_dict,orient='index')
    sparse_build_time_df.plot.bar(ax=ax_sparse,rot=45,legend=False,)
    dense_build_time_df.plot.bar(ax=ax_dense,rot=45,legend=False)

    
    fpath = FPATH[:10] + dataset + '_update_time'
    fig.savefig(fpath)
    out_fname = OUT_DIR +  dataset + '_update_time.tex'
    tikzplotlib.save(out_fname)

def error_iterations_plot(dataset, feature_hyper):
    """
    Plots the build times
    """
    file = FPATH + dataset + '.json'
    with open(file) as json_file:
        data_dict = json.load(json_file)
        
    dense_build_time_dict = {}
    sparse_build_time_dict = {}
    fig, ax = plt.subplots(dpi=150)
    
    for k,v in data_dict.items():
        if k == 'Exact':
             continue
        error = np.squeeze(np.mean(data_dict[k]['errors'][feature_hyper],axis=1))
        _plot_params = fd_real_data_plot_config[k]
        _plot_params['label'] = k
        ax.plot(error,**_plot_params)
        
    ax.legend(frameon=False)
    ax.set_ylim(1E-15,10.)
    ax.set_yscale('log',base=10)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Relative Error')

    fpath = FPATH[:10] + dataset + '_error_iterations'
    fig.savefig(fpath)
    out_fname = OUT_DIR +  dataset + '_error_iterations.tex'
    tikzplotlib.save(out_fname)


def error_time_plot(dataset,feature_hyper):
    """
    Plots the build times
    """
    file = FPATH + dataset + '.json'
    with open(file) as json_file:
        data_dict = json.load(json_file)
        
    dense_build_time_dict = {}
    sparse_build_time_dict = {}
    fig, ax = plt.subplots(dpi=150)
    
    for k,v in data_dict.items():
        if k == 'Exact':
             continue
        error = np.squeeze(np.mean(data_dict[k]['errors'][feature_hyper],axis=1))
        times = np.squeeze(np.mean(data_dict[k]['all_times'][feature_hyper],axis=1))
        _plot_params = fd_real_data_plot_config[k]
        _plot_params['label'] = k
        ax.plot(times, error,**_plot_params)
        
    ax.legend(frameon=False)
    ax.set_ylim(1E-15,10.)
    ax.set_yscale('log',base=10)
    ax.set_xlabel('Time')
    ax.set_ylabel('Relative Error')

    fpath = FPATH[:10] + dataset + '_error_time'
    fig.savefig(fpath)
    out_fname = OUT_DIR +  dataset + '_error_time.tex'
    tikzplotlib.save(out_fname)

def error_iterations_time(dataset,feature_hyper):
    file = FPATH + dataset + '.json'
    with open(file) as json_file:
        data_dict = json.load(json_file)
    
    fig, axes = plt.subplots(dpi=150,ncols=2)
    ax_iters, ax_time = axes
    for k,v in data_dict.items():
        if k == 'Exact':
             continue
        error = np.squeeze(np.mean(data_dict[k]['errors'][feature_hyper],axis=1))
        times = np.squeeze(np.mean(data_dict[k]['all_times'][feature_hyper],axis=1))
        _plot_params = fd_real_data_plot_config[k]
        #_plot_params['label'] = k
        ax_iters.plot(error,**_plot_params)
        ax_time.plot(times,error,**_plot_params)
        
    #ax_iters.legend(frameon=False)
    ax_iters.set_ylim(1E-15,10.)
    ax_iters.set_yscale('log',base=10)
    ax_iters.set_xlabel('Iterations')
    ax_iters.set_ylabel('Relative Error')
    ax_iters.grid()
    
    ax_time.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),frameon=False,ncol=2)
    ax_time.set_ylim(1E-15,10.)
    ax_time.set_yscale('log',base=10)
    ax_time.set_xlabel('Time')
    ax_time.set_ylabel('Relative Error')
    ax_time.grid()
    fpath = FPATH[:10] + dataset + '_error_profile'
    fig.savefig(fpath)
    out_fname = OUT_DIR +  dataset + '_error_profile.tex'
    tikzplotlib.save(out_fname)



def main():
    
    files_and_hypers = []
    exp_setup = {
        'CoverType' : '2',
        'w8a'       : '2500'
    }

    for i,f in enumerate(exp_setup.items()):
        build_time_plot(f[0], f[1])
        update_time_plot(f[0],f[1])
        error_iterations_plot(f[0],f[1])
        error_time_plot(f[0],f[1])
        error_iterations_time(f[0],f[1])
        


if __name__ == '__main__':
    main()
    