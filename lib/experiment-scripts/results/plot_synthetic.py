import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tikzplotlib
from plot_config import ihs_plot_params, classical_plot_params

OUT_DIR = 'synthetic/'

def coeffs_vs_n(df,fname):
    """
    Generates the error vs iterations profile for the dataframe df

    This is the plot for experiment 0
    """

    cols = ['Optimal', 'IHS CountSketch', 'Classical CountSketch', 'Classical SRHT',  'Classical SJLT',  'Classical Gaussian']
    #fig,axes = plt.subplots(dpi=150)
    fig,ax_c = plt.subplots(dpi=150)
    
    #ax_c, ax_ihs = axes
    x = df['Rows']
    for col in cols: #set(df.columns).difference(drop_rows):
        plot_kwargs = classical_plot_params[col]
        #if not in drop_ihs: #col != "Rows":
        plot_kwargs['label'] = col
        ax_c.plot(x,df[col],**plot_kwargs)
        
    ax_c.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3),
    fancybox=False, shadow=False, ncol=2,frameon=False)
    ax_c.set_yscale('log',base=10)
    ax_c.set_xscale('log',base=10)
    ax_c.set_ylabel('Coefficient Error to Model')
    ax_c.set_xlabel('Number of rows')
    #ax_c.legend(bbox_to_anchor=(1.05, 0.95),frameon=False)
    ax_c.grid()
    # Strip the .csv for file name
    out_fname = OUT_DIR + fname[:-4] + '-model-error-n.tex'
    tikzplotlib.save(out_fname)

def coeffs_error_opt(df,fname):
    """
    This is the plot for experiment 1a
    """
    fig,ax = plt.subplots(dpi=150)
    x = df.index
    for col in df.columns:
        if col[-1] == '5':
            plot_kwargs = ihs_plot_params[col[:-1]]
            plot_kwargs['markersize'] =  3.0 #0.5*ihs_plot_params[col[:-1]]['markersize']
        else:
            plot_kwargs = ihs_plot_params[col[:-2]]
            plot_kwargs['markersize'] = 5.0 
        plot_kwargs['label'] = col
        ax.plot(1+x,df[col],**plot_kwargs)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
    fancybox=False, shadow=False, ncol=4,frameon=False)
    ax.set_yscale('log',base=10)
    ax.set_xticks(np.arange(5, 25, 5))
    #ax.set_xscale('log',base=10)
    ax.set_ylabel('Coeff Error Opt')
    #ax.set_ylim(1E-3,1)
    ax.grid()
    out_fname = fname[:-4] + '-error-iterations.tex'
    tikzplotlib.save(out_fname)
    

def coeffs_error_model(df,fname):
    """
    Plot Log of model error to sketched solutions in base10 logarithm.

    This is the plot for experiment 1b
    """
    d_over_n = np.sqrt(200 / 6000)
    fig,ax = plt.subplots(dpi=150)
    x = df.index
    for col in df.columns:
        if col[-1] == '5':
            plot_kwargs = ihs_plot_params[col[:-1]]
            plot_kwargs['markersize'] =  3.0 
        else:
            plot_kwargs = ihs_plot_params[col[:-2]]
            plot_kwargs['markersize'] = 5.0 
        plot_kwargs['label'] = col
        #ax.plot(1+x,df[col],**plot_kwargs)
        ax.plot(1+x,np.log10(df[col]),**plot_kwargs)
    ax.plot(1+x,np.log10(d_over_n*np.ones_like(1+x)),**classical_plot_params['Optimal'])
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3),
    fancybox=False, shadow=False, ncol=3,frameon=False)
    
    ax.set_ylabel('Log Error to model')
    ax.set_xlabel('Iterations')
    ax.set_ylabel(r'$|| \hat{x} - x^+ ||_A$')
    ax.set_xlim(0.5,10)
    ax.grid()
    
    #ax.plot(1+x,(d_over_n*np.ones_like(1+x)),**classical_plot_params['Optimal'])
    #ax.set_yscale('log',base=10)
    
    out_fname = fname[:-4] + '-error-iterations.tex'
    tikzplotlib.save(out_fname)


def main():
    files = [
        'experiment0-ihs-ols.csv',   
        'experiment1-ihs-iterations-opt.csv',
        'experiment1-ihs-iterations-model.csv',
    ]

    for i,f in enumerate(files):
        df = pd.read_csv(f,header=[0])
        if i == 0:
            coeffs_vs_n(df,f)
        elif i == 1:
            coeffs_error_opt(df,f)
        elif i == 2:
            print(df.head())
            coeffs_error_model(df,f)
        print(i,f)
if __name__ == '__main__':
    main()