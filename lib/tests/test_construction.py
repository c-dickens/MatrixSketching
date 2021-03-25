import sys
sys.path.append('../',)
import pytest
import numpy as np
from gaussian_sketch import GaussianSketch
from count_sketch import CountSketch
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
    X = np.random.randn(1000,50)
    return X


def test_summary_size(data_to_test):
    '''
    Tests that the summary returned has number of rows equal
    to the required projectiont dimension'''
    sketch_dim = 100
    n,d = data_to_test.shape
    sketches = {
        'gauss' : GaussianSketch(sketch_dim,n,d),
        'countsketch' : CountSketch(sketch_dim,n,d)
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