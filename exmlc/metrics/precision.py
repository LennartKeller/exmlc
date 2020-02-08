import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

from .utils import top_n_idx_sparse


def sparse_average_precision_at_k(y_true: csr_matrix, y_scores: csr_matrix, k: int = 5) -> float:
    """
    Computes the average precision at k for sparse binary matrices.
    :param y_true: grounded truth in binary format (n_samples, n_labels)
    :param y_scores: predictions in respresentation that can be ranked (e.g. probabilities)
    :param k: top k labels to check
    :return: precision at k score
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


def precision_at_k(y_true: csr_matrix, y_scores: csr_matrix, k=3) -> float:
    if y_true.shape != y_scores.shape:
        raise Exception

    if y_true.shape[0] > 1:
        raise Exception

    top_idx = top_n_idx_sparse(y_scores, n=k)[0]
    s = 0
    for ind in top_idx:
        if y_true[0, ind] > 0:
            s += 1
    return s / k


def average_precision_at_k(y_true: csr_matrix, y_scores: csr_matrix, k=5) -> float:
    precisions = []
    for true_row, pred_row in zip(y_true, y_scores):
        precisions.append(precision_at_k(true_row, pred_row, k=k))
    return np.mean(precisions)
