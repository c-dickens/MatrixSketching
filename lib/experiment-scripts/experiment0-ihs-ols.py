import numpy as np

def experimental_data(n,d, sigma=1.0,seed=100):
    """
    The data for this experiment is taken from 3.1 https://jmlr.org/papers/volume17/14-460/14-460.pdf
    A       : n \times d matrix of N(0,1) entries
    x_model : d \times 1 vector chosen uniformly from the d dimensional sphere by choosing 
              random normals and then normalising.
    w       : Vector of noise perturbation which is N(0,sigma**2*I_n)
    y = A@x_model + w
    """
    np.random.seed(seed)
    A = np.random.randn(n,d)
    x_model = np.random.randn(d,1)
    x_model /= np.linalg.norm(x_model,axis=0)
    w = np.random.multivariate_normal(mean=np.zeros((n,)),cov=sigma**2*np.eye(n),size=(n,))
    y = A@x_model + w
    return y,A,x_model

def main():
    """
    Task: IHS-OLS with random projections, specifically, the CountSketch

    Figure 1 https://jmlr.org/papers/volume17/14-460/14-460.pdf
    """
    n = 1000
    d = 10
    y, A, x_model = experimental_data(n,d)
    print(y.shape, A.shape)

if __name__ == '__main__':
    main()