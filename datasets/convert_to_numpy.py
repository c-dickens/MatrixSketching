"""
Converts the datasets from various formats to numpy.
Data is read in and saved into [X,y] format with X standardised.

Prior to executing this one should run:
wget https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz
gunzip covtype.data.gz
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w8a
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

datasets = {
    'covertype' : {
        'path' : 'covtype.data',
        'url'  : 'https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz',
        'target' : -1
        },

    'w8a' : {
        'path' : 'w8a',
        'url'  : 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w8a',
    }   

    } 



def main():
    from sklearn.datasets import load_svmlight_file
    for d in datasets:
        print('INSPECTING DATASET ', d)
        try:
            # read the numpy file -- Numpy file exists
            dat = np.load(d+'.npy')
            print(f'{d} data already in numpy format.')
        except:
            # Numpy file does not exists so load via pandas
            if d in ['w8a']:
                X,y =  load_svmlight_file('w8a')
                X = X.toarray()
                df_arr = np.c_[X,y]
            else:
                print(f'{d} : obtaining .npy from input format')
                data_path = datasets[d]['path']
                df = pd.read_csv(data_path, header=0)
                print(df.head())
                df_arr = df.to_numpy()

            # Save the data in format data.npy
            df_arr = StandardScaler().fit_transform(df_arr)
            print(df_arr.shape)
            fname = d+'.npy'
            np.save(fname, df_arr)

if __name__ == '__main__':
    main()