import numpy as np
from scipy.sparse import csr_matrix


def top_n_idx_sparse(matrix: csr_matrix, n: int) -> np.ndarray:
    """
    Return a matrix with of shape (matrix.shape[0], n) with n indices of the n largest value of each row in the matrix
    The indices of the n top values are not sorted itself.
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


def get_n_top_values_sorted(matrix: csr_matrix, n: int, decsending: bool = True) -> np.ndarray:
    """
    Returns a array of arrays with shape (matrix.shape[0],) where each array contains the sorted indices
    of the top n values of the corresponding row.
    :param matrix:
    :param n:
    :param decsending:
    :return:
    """
    sorted_top_n_idx = []
    for row in matrix:
        row = row.toarray().ravel()
        n_norm = min(np.count_nonzero(row), n)
        ind = np.argpartition(row, -n_norm)[-n_norm:]
        ind = ind[np.argsort(row[ind])]
        if decsending:
            ind = np.flip(ind)
        sorted_top_n_idx.append(ind)
    return np.array(sorted_top_n_idx)


if __name__ == '__main__':
    m = csr_matrix(
        [
            [1, 5, 7, 9, 0, 0, 0],
            [1, 0, 0, 0, 2, 0, 3],
            [0, 0, 0, 0, 0, 0, 3],
        ]
    )
    print(top_n_idx_sparse(m, n=2))
