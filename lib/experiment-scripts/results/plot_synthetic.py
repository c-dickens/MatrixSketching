import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tikzplotlib
from plot_config import ihs_plot_params

OUT_DIR = 'synthetic/'


def main():
    files = [
        'experiment0-ihs-ols.csv',   
        'experiment1-ihs-iterations-opt.csv',
        'experiment1-ihs-iterations-model.csv',
    ]

    for f in files:
        df = pd.read_csv(f,header=[0,1])
        coeffs_vs_iterations(df,f)
        coeffs_vs_wall_time(df,f)
        test_vs_iterations(df,f)
        test_vs_wall_time(df,f) 

if __name__ == '__main__':
    main()