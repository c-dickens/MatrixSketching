import numpy as np
from scipy import linalg
from matrix_sketch import MatrixSketch

class FrequentDirections(MatrixSketch):
    def __init__(self, sketch_dim:int, n_data_rows:int, \
        n_data_cols:int):
        """
        Class wrapper for all FD-type methods

        __rotate_and_reduce__ is not defined for the standard FrequentDirections but is for the
        subsequent subclasses which inherit from FrequentDirections.
        """
        super(FrequentDirections,self).__init__(sketch_dim)
        super(FrequentDirections,self)._prepare_sketch(n_data_rows, n_data_cols)
        # self.n = n_data_rows
        self.d = n_data_cols
        self.delta = 0.  # For RFD

        if sketch_dim is not None:
            self.sketch_dim = sketch_dim
        self.sketch_matrix = np.zeros((self.sketch_dim, self.d), dtype=float)
        self.Vt = np.zeros((self.sketch_dim, self.d), dtype=float)
        self.sigma_squared = np.zeros(self.sketch_dim, dtype=float)

    def sketch(self, X,batch_size):
        """
        Fits the FD transform to dataset X
        """
        n = X.shape[0]
        aux = np.empty((self.sketch_dim+batch_size, self.d))
        for i in range(0, n, batch_size):
            batch = X[i:i + batch_size, :]
            #aux = np.concatenate((self.sketch, batch), axis=0)
            aux[0:self.sketch_dim, :] = self.sketch_matrix
            aux[self.sketch_dim:self.sketch_dim+batch.shape[0], :] = batch
            # ! WARNING - SCIPY SEEMS MORE ROBUST THAN NUMPY SO COMMENTING THIS WHICH IS FASTER OVERALL
            # try:
            #     _, s, self.Vt = np.linalg.svd(aux, full_matrices=False)
            # except np.linalg.LinAlgError:
            #     _, s, self.Vt = linalg.svd(aux, full_matrices=False, lapack_driver='gesvd')
            _, s, self.Vt = linalg.svd(aux, full_matrices=False, lapack_driver='gesdd')
            self.sigma_squared = s ** 2
            self.__rotate_and_reduce__()
            self.sketch_matrix = self.Vt * np.sqrt(self.sigma_squared).reshape(-1, 1)

    def get_fd_outputs(self):
        return self.sketch, self.sigma_squared, self.Vt, self.delta


class FastFrequentDirections(FrequentDirections):
    """
    Implements the fast version of FD by doubling space
    """

    def __rotate_and_reduce__(self):
        self.sigma_squared = self.sigma_squared[:self.sketch_dim] - self.sigma_squared[self.sketch_dim]
        self.Vt = self.Vt[:self.sketch_dim]


class RobustFrequentDirections(FrequentDirections):
    """
    Implements the RFD version of FD by maintaining counter self.delta.
    Still operates in the `fast` regimen by doubling space, as in
    FastFrequentDirections
    """

    def __rotate_and_reduce__(self):
        self.delta += self.sigma_squared[self.sketch_dim] / 2.
        self.sigma_squared = self.sigma_squared[:self.sketch_dim] - self.sigma_squared[self.sketch_dim]
        self.Vt = self.Vt[:self.sketch_dim]

# import numpy as np
# from scipy import linalg
# from matrix_sketch import MatrixSketch

# class FrequentDirections(MatrixSketch):
#     def __init__(self, sketch_dim:int, n_data_rows:int, n_data_cols:int,seed=None):
#         """
#         Class wrapper for all FD-type methods
#         __rotate_and_reduce__ is not defined for the standard FrequentDirections but is for the
#         subsequent subclasses which inherit from FrequentDirections.

#         seed is a placeholder for consistency with randomised methods.
#         """
#         super(FrequentDirections,self).__init__(sketch_dim)
#         super(FrequentDirections,self)._prepare_sketch(n_data_rows, n_data_cols)
#         self.d = n_data_cols
#         self.delta = 0.  # For RFD

#         if sketch_dim is not None:
#             self.sketch_dim = sketch_dim
#         self.sketch_matrix = np.zeros((self.sketch_dim, self.d), dtype=float)
#         self.Vt = np.zeros((self.sketch_dim, self.d), dtype=float)
#         self.sigma_squared = np.zeros(self.sketch_dim, dtype=float)

#     def sketch(self, X, batch_size=1):
#         """
#         Fits the FD transform to dataset X.
#         """
#         n = X.shape[0]
#         aux = np.empty((self.sketch_dim+batch_size, self.d))
#         for i in range(0, n, batch_size):
#             batch = X[i:i + batch_size, :]
#             #aux = np.concatenate((self.sketch, batch), axis=0)
#             aux[0:self.sketch_dim, :] = self.sketch_matrix
#             aux[self.sketch_dim:self.sketch_dim+batch.shape[0], :] = batch
#             # ! WARNING - SCIPY SEEMS MORE ROBUST THAN NUMPY SO COMMENTING THIS WHICH IS FASTER OVERALL
#             # try:
#             #     _, s, self.Vt = np.linalg.svd(aux, full_matrices=False)
#             # except np.linalg.LinAlgError:
#             #     _, s, self.Vt = linalg.svd(aux, full_matrices=False, lapack_driver='gesvd')
#             _, s, self.Vt = linalg.svd(aux, full_matrices=False, lapack_driver='gesdd')
#             self.sigma_squared = s ** 2
#             self.__rotate_and_reduce__()
#             self.sketch_matrix = self.Vt * np.sqrt(self.sigma_squared).reshape(-1, 1)

#     def get_all_fd(self):
#         """
#         Returns all of the objects computed in FD.
#         """
#         return self.sketch_matrix, self.sigma_squared, self.Vt, self.delta


# class FastFrequentDirections(FrequentDirections):
#     """
#     Implements the fast version of FD by doubling space
#     """

#     def __rotate_and_reduce__(self):
#         self.sigma_squared = self.sigma_squared[:self.sketch_dim] - self.sigma_squared[self.sketch_dim]
#         self.Vt = self.Vt[:self.sketch_dim]


# class RobustFrequentDirections(FrequentDirections):
#     """
#     Implements the RFD version of FD by maintaining counter self.delta.
#     Still operates in the `fast` regime by doubling space, as in
#     FastFrequentDirections
#     """

#     def __rotate_and_reduce__(self):
#         self.delta += self.sigma_squared[self.sketch_dim] / 2.
#         self.sigma_squared = self.sigma_squared[:self.sketch_dim] - self.sigma_squared[self.sketch_dim]
#         self.Vt = self.Vt[:self.sketch_dim]