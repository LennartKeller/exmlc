import numpy as np
from scipy.sparse import csr_matrix


def top_n_idx_sparse(matrix: csr_matrix, n: int) -> np.ndarray:
    """
    Return a matrix with of shape (matrix.shape[0], n) with n indices of the n largest value of each row in the matrix
    The indices of the n top values are not sorted itself.
    NOTE: If there are less than n values
    :param matrix:
    :param n:
    :return:
    """

    top_n_idx = []
    for le, ri in zip(matrix.indptr[:-1], matrix.indptr[1:]):
        n_row_pick = min(n, ri - le)
        top_idx = matrix.indices[le + np.argpartition(matrix.data[le:ri], -n_row_pick)[-n_row_pick:]]
        top_n_idx.append(top_idx)
    return np.array(top_n_idx)