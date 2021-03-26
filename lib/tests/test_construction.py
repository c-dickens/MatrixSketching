import sys
sys.path.append('../',)
import pytest
import numpy as np
from gaussian_sketch import GaussianSketch
from count_sketch import CountSketch
from sparse_jlt import SparseJLT
from srht_sketch import SRHTSketch
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
# https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility

@pytest.fixture
def data_to_test():
    '''
    Generates a sample of test data for the tests.
    This can be replaced by pulling a real dataset.
    To ensure the data-target test passes for the SRHT
    then length of the input should be a power of 2.
    '''
    np.random.seed(100)
    #X = np.random.randn(20,5)
    #X = 2*np.eye(10,20)[:,:5]
    n,d = 1000,5
    X = np.random.randn(n,d) #/ np.random.randn(n,d)
    #X /= np.linalg.norm(X,axis=0)
    #true_norm = np.linalg.norm(X,ord='fro')**2
    return X


def test_summary_size(data_to_test):
    '''
    Tests that the summary returned has number of rows equal
    to the required projectiont dimension'''
    sketch_dim = 500
    n,d = data_to_test.shape
    true_norm = np.linalg.norm(data_to_test,ord='fro')**2
    sketches = {
        'gauss' : GaussianSketch(sketch_dim,n,d),
        'srht'  : SRHTSketch(sketch_dim,n,d),
        # 'countsketch' : CountSketch(sketch_dim,n,d),
        # 'sjltsketch' : SparseJLT(sketch_dim,n,d,col_sparsity=20)
    }
    for sk_name, sk_method in sketches.items():
        g = sk_method
        g.sketch(data_to_test)
        summary = g.get()
        summary_u, summary_sig, summary_vt = g.get(in_svd=True)
        assert summary.shape == (sketch_dim,d)
        assert summary_u.shape == (sketch_dim,d)
        assert summary_sig.shape == (d,)
        assert summary_vt.shape == (d,d)
        sk_norm = np.linalg.norm(summary,ord='fro')**2
        err = np.abs(sk_norm - true_norm) / true_norm
        print(f'{sk_name}:\t{err:.5f}')