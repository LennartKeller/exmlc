import numpy as np
from scipy.sparse import csr_matrix, lil_matrix


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


def sparse_average_precision_at_k(y_true: csr_matrix, y_scores: csr_matrix, k: int = 5) -> float:
    """
    TODO docs
    :param y_true:
    :param y_scores:
    :param k:
    :return:
    """
    if y_true.shape != y_scores.shape:
        raise Exception('y_true and y_pred must have same shape')
    if y_true.shape[1] < k:
        raise Exception('Less labels than k')

    # get indices of k top values of y_pred
    top_idx = top_n_idx_sparse(y_scores, k)
    # create new matrix with shape == y_true.shape with only top ranked labels
    y_pred_binary_only_top = lil_matrix(y_true.shape, dtype='int8')
    for index, (binary_row, idx_row) in enumerate(zip(y_pred_binary_only_top, top_idx)):
        y_pred_binary_only_top[index, idx_row] = 1
    y_pred_binary_only_top = y_pred_binary_only_top.tocsr()
    # compute precision

    # get correct predicted labels
    correct_labelled = y_true.multiply(y_pred_binary_only_top)
    summed_precision = []

    for index, (row, score_row) in enumerate(zip(correct_labelled, y_scores)):
        # check special case that corresponding y_true row is empty => unlabeled instance
        if y_true[index].count_nonzero() == 0:
            # if no labels where predicted add 1 to sum
            if score_row.count_nonzero() == 0:
                summed_precision.append(1.0)
            else:
                summed_precision.append(0)
        else:
            summed_precision.append(row.count_nonzero() / k)

    return sum(summed_precision) / len(summed_precision)
