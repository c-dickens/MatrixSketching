from math import sqrt as m_sqrt
import numpy as np
from scipy.fftpack import dct
from scipy.linalg import hadamard
from matrix_sketch import MatrixSketch

class SRHTSketch(MatrixSketch):
    """
    Class wrapper for the SRHT Sketch
    """
    def __init__(self,sketch_dim:int, n_data_rows:int, \
        n_data_cols:int):

        super(SRHTSketch,self).__init__(sketch_dim)
        super(SRHTSketch,self)._prepare_sketch(n_data_rows, n_data_cols)

    # def _sample(self,seed=100):
    #     """
    #     Samples the random linear transform S with 'seed'
    #     S is a Gaussian random transform.
    #     """
    #     np.random.seed(seed)
    #     diag = np.random.choice([1.,-1.], self.new_n)[:,None]
    #     S /= np.sqrt(self.sketch_dim)
    #     return S

    def sketch(self,mat,seed=100):
        """
        We implement DCT for now, note that the columns need normalising 
        after the transform to ensure orthogonality.
        cf. https://fairyonice.github.io/2D-DCT.html
        NOTE THAT the factor of 1/sqrt(n) is already absorbed into the transform in scipy (unlike their 
        hadamard matrix).
        """
        # np.random.seed(seed)
        # np.set_printoptions(precision=4)
        # diag = np.random.choice([1.,-1.], self.n_data_rows)[:,None]
        # 
        # HDA = dct(diag*mat)
        # # ! I don/t understand this normalisation yet but it looks right 
        # # Divide out column norms to make the transform unitary unsure about rescaling by root(n)
        # HDA = HDA * (m_sqrt(self.n_data_rows)/np.linalg.norm(HDA,axis=0)) 
        # sample = np.random.choice(self.n_data_rows, self.sketch_dim, replace=False)
        # self.sketch_matrix = HDA[sample]* (1./(self.sketch_dim)**0.5) 
        diag = np.random.choice([1.,-1.], self.n_data_rows)[:,None]
        DA = diag*mat
        HDA = dct(DA,axis=0,norm='ortho') * m_sqrt(self.n_data_rows)
        #H = hadamard(self.n_data_rows) #/ m_sqrt(self.n_data_rows)
        #print(H)
        #HDA = H@(diag*mat)
        #HDA = HDA / np.linalg.norm(HDA,axis=0) # Divide out column norms to make the transform unitary
        sample = np.random.choice(self.n_data_rows, self.sketch_dim, replace=False)
        # ! I don/t understand this normalisation yet but it looks right 
        #self.sketch_matrix = HDA[sample]* (self.n_data_rows/(self.sketch_dim)**0.5) 
        self.sketch_matrix = HDA[sample,:]*(1./(self.sketch_dim)**0.5) 



        