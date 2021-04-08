import openml
from sklearn.datasets import fetch_openml
import os
import scipy.sparse as sparse
from sklearn.preprocessing import Normalizer

all_datasets = {
    "covertype" : {
    "url" : 'https://www.openml.org/d/293',
    "outputFileName" : 'covertype',
    'input_destination' : 'OPENML',
    'openml_id' : 293,
    'sparse_format' : True
    },
}


def get_openml_data(dataset):
    '''Tool to downloads openml data and put into correct format'''
    out_file = all_datasets[dataset]['outputFileName']
    _id = all_datasets[dataset]['openml_id']
    #dataset = openml.datasets.get_dataset(_id)

    # Print a summary
    # print("This is dataset '%s', the target feature is '%s'" %
    #       (dataset.name, dataset.default_target_attribute))
    # X, y, categorical_indicator, attribute_names = dataset.get_data(
    # target=dataset.default_target_attribute
    # )
    # print(type(X),X.shape)
    dat = fetch_openml(data_id=_id)
    X = dat['data']
    y = dat['target'].astype('float')

    if isinstance(X,sparse.csr_matrix):
        nnz = X.count_nonzero()
        #X_coo = X.tocoo
        y_coo = sparse.coo_matrix(y).transpose()
        n,d = X.shape
        # print(f'Shape of data: {n},{d}')
        # print(f'Aspect ratio: {d/n}')
        # print(f'Type of data: {type(X)},{type(y)}')
        X_norm = Normalizer().fit_transform(X)
        X_norm_coo = X_norm.tocoo()
        print(X_norm_coo.shape, y_coo.shape)
        Xy = sparse.hstack([X_norm_coo,y_coo])
        print(Xy.shape)
        print(type(Xy))
        sparse.save_npz(out_file,Xy)
    # else:
    #     nnz = np.count_nonzero(X)
    #     data = np.c_[X,y]
    #     data[np.isnan(data)] = 0
    #     np.save(out_file, data)

    # print(f'Density: {nnz/(n*d)}')

def main():
    '''
    Download data from all_datasets
    '''



    for dataset in all_datasets:
        print(dataset)
        out_file = all_datasets[dataset]['outputFileName']
        file_url = all_datasets[dataset]['url']

        # if the .npy version doesn't exist then make one

        if not (os.path.isfile(out_file + '.npy') or os.path.isfile(out_file + '.npz')):    
            get_openml_data(dataset)

if __name__ == '__main__':
    main()

