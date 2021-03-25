import sys
sys.path.append('../',)
import pytest
import numpy as np
from gaussian_sketch import GaussianSketch


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
    g = GaussianSketch(sketch_dim,n,d)
    g.sketch_matrix(data_to_test)
    summary = g.get()
    summary_u, summary_sig, summary_vt = g.get(in_svd=True)
    assert summary.shape == (sketch_dim,d)
    assert summary_u.shape == (sketch_dim,d)
    assert summary_sig.shape == (d,)
    assert summary_vt.shape == (d,d)