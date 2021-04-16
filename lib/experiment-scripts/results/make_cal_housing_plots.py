import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tikzplotlib
from plot_config import ihs_plot_params

OUT_DIR = 'cal_housing_ihs/'

def coeffs_vs_iterations(df,fname):
    """
    Generates the error vs iterations profile for the dataframe df
    """
    fig, ax = plt.subplots(dpi=150)
    iterations = 1 + df.index
    for c in set(df.columns.get_level_values(0)):
        if c != 'Exact(SVD)':
            plot_kwargs = ihs_plot_params[c]
            ax.plot(iterations, df[c,'Coefficient Error'],label=c,**plot_kwargs)
    ax.grid()
    ax.set_yscale('log')
    ax.set_ylabel('Log Coefficient Error')
    ax.set_xlabel('Iterations')
    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
    fancybox=False, shadow=False, ncol=3,frameon=False)
    # Strip the .csv for file name
    out_fname = OUT_DIR + fname[:-4] + '-error-iterations.tex'
    tikzplotlib.save(out_fname)
    
def coeffs_vs_wall_time(df,fname):
    """
    Generates the error vs iterations profile for the dataframe df
    """
    fig, ax = plt.subplots(dpi=150)
    iterations = 1 + df.index
    for c in set(df.columns.get_level_values(0)):
        total_time = 0.
        if c != 'Exact(SVD)':
            plot_kwargs = ihs_plot_params[c]
            iter_time = df[c]['Sketch'] + df[c]['SVD'] + df[c]['Solve']
            if c == 'Classical':
                ax.plot(iter_time.cumsum()[0],df[c,'Coefficient Error'][0],label=c,**plot_kwargs)
            else:
                ax.plot(iter_time.cumsum(),df[c,'Coefficient Error'],label=c,**plot_kwargs)
    ax.axvline(x=df['Exact(SVD)', 'SVD'].iloc[0],color='black',linestyle=(0, (5, 1)),label='SVD')
    ax.grid()
    ax.set_yscale('log')
    ax.set_xscale('log',base=10)
    ax.set_ylabel('Log Coefficient Error')
    ax.set_xlabel('Log (Time (Seconds))')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
    fancybox=False, shadow=False, ncol=3,frameon=False)
    # Strip the .csv for file name
    out_fname = OUT_DIR + fname[:-4] + '-error-wall-time.tex'
    tikzplotlib.save(out_fname)

    
def test_vs_iterations(df,fname):
    """
    Generates the test erro vs number of iterations for dataframe df
    """
    fig, ax = plt.subplots(dpi=150)
    iterations = 1 + df.index
    opt_test_error = df['Exact(SVD)']['Test Error'][0]
    for c in set(df.columns.get_level_values(0)):
        if c != 'Exact(SVD)':
            plot_kwargs = ihs_plot_params[c]
            test_err_ratio = np.abs(df[c,'Test Error']/opt_test_error - 1)
            ax.plot(iterations, test_err_ratio,label=c,**plot_kwargs)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
    fancybox=False, shadow=False, ncol=3,frameon=False)
    ax.grid()
    ax.set_yscale('log')
    ax.set_ylabel('Log Test Error')
    ax.set_xlabel('Iterations')
    # Strip the .csv for file name
    out_fname = OUT_DIR + fname[:-4] + 'test-error-iterations.tex'
    tikzplotlib.save(out_fname)
    
def test_vs_wall_time(df,fname):
    """
    Generates the test erro vs number of iterations for dataframe df
    """
    fig, ax = plt.subplots(dpi=150)
    iterations = 1 + df.index
    opt_test_error = df['Exact(SVD)']['Test Error'][0]
    for c in set(df.columns.get_level_values(0)):
        if c != 'Exact(SVD)':
            plot_kwargs = ihs_plot_params[c]
            test_err_ratio = np.abs(df[c,'Test Error']/opt_test_error - 1)
            iter_time = df[c]['Sketch'] + df[c]['SVD'] + df[c]['Solve']
            if c == 'Classical':
                ax.plot(iter_time.cumsum()[0],test_err_ratio[0],label=c,**plot_kwargs)
            else:
                ax.plot(iter_time.cumsum(),test_err_ratio,label=c,**plot_kwargs)
            #ax.plot(iter_time.cumsum(), test_err_ratio,label=c,**plot_kwargs)
    ax.axvline(x=df['Exact(SVD)', 'SVD'].iloc[0],color='black',linestyle=':',label='SVD')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
    fancybox=False, shadow=False, ncol=3,frameon=False)
    ax.grid()
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel('Log Test Error')
    ax.set_xlabel('Log (Time (Seconds))')
    # Strip the .csv for file name
    out_fname = OUT_DIR + fname[:-4] + 'test-error-wall-time.tex'
    tikzplotlib.save(out_fname)

def all_error_time(files):
    """
    Generates the 3 x 1 plot of all wall clock times plots
    """

    fig=plt.figure(dpi=150)
    ax1 = plt.subplot(311)
    ax2 = plt.subplot(312)
    ax3 = plt.subplot(313)
    #plt.setp(ax2.get_yticklabels()[0], visible=False)
    ax1.get_shared_x_axes().join(ax1, ax2)
    ax3.get_shared_x_axes().join(ax1, ax2, ax3)
    ax1.set_xticklabels([])
    ax2.set_xticklabels([])
    # ax2.autoscale() ## call autoscale if needed
    fig.subplots_adjust(hspace=0.1)
    
    axes = [ax1, ax2, ax3]
    for i,f in enumerate(files):
        df = pd.read_csv(f,header=[0,1])
        ax = axes[i]
        for c in set(df.columns.get_level_values(0)):
            total_time = 0.
            if c != 'Exact(SVD)':
                plot_kwargs = ihs_plot_params[c]
                iter_time = df[c]['Sketch'] + df[c]['SVD'] + df[c]['Solve']
                if c == 'Classical':
                    ax.plot(iter_time.cumsum()[0],df[c,'Coefficient Error'][0],label=c,**plot_kwargs)
                else:
                    ax.plot(iter_time.cumsum(),df[c,'Coefficient Error'],label=c,**plot_kwargs)
    # Formatting for all axes
    for ax in axes:
        ax.set_ylim(1E-6,1E-3)
        ax.set_yscale('log',base=10)
        ax.set_xscale('log',base=2)
        ax.grid()
        ax.axvline(x=df['Exact(SVD)', 'SVD'].iloc[0],color='black',linestyle=(0, (5, 1)),label='SVD')
        ax.set_ylabel('Log Coefficient Error')
    ax3.set_xlabel('Log (Time (seconds))')
    
    # Legend:
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.5),
    fancybox=False, shadow=False, ncol=3,frameon=False) 
    out_fname = 'cal_housing_all_wall_clock_times.tex'
    tikzplotlib.save(out_fname)

def test_error_vs_iterations_time(df,fname):
    fig, axes = plt.subplots(dpi=150,ncols=2,gridspec_kw={'wspace':0.0})
    iterations = 1 + df.index
    ax_iters, ax_time = axes
    opt_test_error = df['Exact(SVD)']['Test Error'][0]
    # Plotting for iterations 
    for c in set(df.columns.get_level_values(0)):
        if c != 'Exact(SVD)':
            plot_kwargs = ihs_plot_params[c]
            test_err_ratio = np.abs(df[c,'Test Error']/opt_test_error)
            ax_iters.plot(iterations, test_err_ratio,label=c,**plot_kwargs)
    
    # Plotting for time
    for c in set(df.columns.get_level_values(0)):
        if c != 'Exact(SVD)':
            plot_kwargs = ihs_plot_params[c]
            test_err_ratio = np.abs(df[c,'Test Error']/opt_test_error)
            iter_time = df[c]['Sketch'] + df[c]['SVD'] + df[c]['Solve']
            if c == 'Classical':
                ax_time.plot(iter_time.cumsum()[0],test_err_ratio[0],label=c,**plot_kwargs)
            else:
                ax_time.plot(iter_time.cumsum(),test_err_ratio,label=c,**plot_kwargs)
        else:
            ax_time.axvline(x=df['Exact(SVD)', 'SVD'].iloc[0],color='black',linestyle=':',label='SVD')
            
    for ax in axes:
        ax.grid()
        #ax.set_ylim(0.9,2.1)
    
    # format ax_iters
    ax_iters.set_ylabel('Test Error ratio')
    ax_iters.set_xlabel('Iterations')
    
    # format ax_time
    
    ax_time.set_xlabel('Log Time')
    ax_time.set_yticklabels([])
    ax_time.set_xscale('log',base=2)
    ax_time.legend(loc='upper center', bbox_to_anchor=(0., 1.2),
    fancybox=False, shadow=False, ncol=3,frameon=False)
    
    out_fname = 'cal_housing_ihs/' + fname[:-4] + 'test-error-iters-wall-time.tex'
    print(out_fname)
    tikzplotlib.save(out_fname)

def main():
    files = [
        'experiment3-cal_housing.csv',
        'experiment3-cal_housing0.125.csv', 
        'experiment3-cal_housing0.25.csv', 
        'experiment3-cal_housing0.5.csv'
    ]
    test_error_vs_iterations_time(pd.read_csv(files[0],header=[0,1]),files[0]) # Don't  want sparsified results
    for f in files:
        df = pd.read_csv(f,header=[0,1])
        coeffs_vs_iterations(df,f)
        coeffs_vs_wall_time(df,f)
        test_vs_iterations(df,f)
        test_vs_wall_time(df,f) 
    all_error_time(files[1:]) # Dont want to plot the non-sparsified data
    

if __name__ == '__main__':
    main()