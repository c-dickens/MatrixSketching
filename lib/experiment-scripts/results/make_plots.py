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

def main():
    files = [
        'experiment3-cal_housing.csv',
        'experiment3-cal_housing0.125.csv', 
        'experiment3-cal_housing0.25.csv', 
        'experiment3-cal_housing0.5.csv'
    ]

    for f in files:
        df = pd.read_csv(f,header=[0,1])
        coeffs_vs_iterations(df,f)
        coeffs_vs_wall_time(df,f)
        test_vs_iterations(df,f)
        test_vs_wall_time(df,f) 

if __name__ == '__main__':
    main()