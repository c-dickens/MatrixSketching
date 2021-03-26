from math import sqrt as m_sqrt
import numpy as np
from scipy.fftpack import dct
from scipy.linalg import hadamard
from matrix_sketch import MatrixSketch
import sys
sys.path.append('/home/dickens/code/fastwht/python')
from hadamard import fastwht

class SRHTSketch(MatrixSketch):
    """
    Class wrapper for the SRHT Sketch.

    fft_transform - 
        'DCT' for Discrete Cosine Transform
        'HAD' for Hadamard transform
    """
    def __init__(self,sketch_dim:int, n_data_rows:int, \
        n_data_cols:int,fft_transform='DCT'):

        super(SRHTSketch,self).__init__(sketch_dim)
        super(SRHTSketch,self)._prepare_sketch(n_data_rows, n_data_cols)
        self.fft_transform = fft_transform
        if self.fft_transform not in ['DCT', 'HAD']:
            raise NotImplementedError(
                'Only Discrete Cosine Transform (DCT) or Hadamard Transform (HAD) implemented'
            )
        # * Preprocessing for Hadamard Transform
        if self.fft_transform == 'HAD':
            self.next_power2_data = self._shift_bit_length(self.n_data_rows)
            self.deficit = self.next_power2_data - self.n_data_rows
            

    def _shift_bit_length(self,x):
        '''
        Given int x find next largest power of 2.
        If x is a power of 2 then x is returned 
        '''
        return 1<<(x-1).bit_length()

    def _hadamard_transform(self,mat):
        new_data = np.concatenate((mat,np.zeros((self.deficit,self.n_data_cols),\
                                dtype=mat.dtype)), axis=0)
        return fastwht(new_data)*self.next_power2_data


    def sketch(self,mat,seed=100):
        """
        We implement DCT for now, note that the columns need normalising 
        after the transform to ensure orthogonality.
        NB. have to take care of how scipy uses FFT see 
        https://stackoverflow.com/questions/62335898/applying-dct-matrix-along-each-axis-not-giving-the-desired-result
        """
        np.random.seed(seed)
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

        if self.fft_transform == 'DCT':
            HDA = dct(DA,axis=0,norm='ortho') * m_sqrt(self.n_data_rows)
        elif self.fft_transform == 'HAD':
            HDA = self._hadamard_transform(DA)
        #H = hadamard(self.n_data_rows) #/ m_sqrt(self.n_data_rows)
        #print(H)
        #HDA = H@(diag*mat)
        #HDA = HDA / np.linalg.norm(HDA,axis=0) # Divide out column norms to make the transform unitary
        sample = np.random.choice(self.n_data_rows, self.sketch_dim, replace=False)
        # ! I don/t understand this normalisation yet but it looks right 
        #self.sketch_matrix = HDA[sample]* (self.n_data_rows/(self.sketch_dim)**0.5) 
        self.sketch_matrix = HDA[sample,:]*(1./(self.sketch_dim)**0.5) 



        