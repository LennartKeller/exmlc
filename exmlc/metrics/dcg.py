from typing import *
from scipy.sparse import csr_matrix
from .utils import top_n_idx_sparse


def average_discounted_cumulative_gain_at_k(y_true: csr_matrix,
                                            y_scores: csr_matrix,
                                            k: int = 3,
                                            normalize: bool = False) -> float:
    """
    TODO
    :param y_true:
    :param y_scores:
    :param k:
    :param normalize:
    :return:
    """

    if normalize:
        raise NotImplementedError

    if y_true.shape != y_scores.shape:
        raise Exception('y_true and y_pred must have same shape')
    if y_true.shape[1] < k:
        raise Exception('Less labels than k')

    k_top_idx = top_n_idx_sparse(y_scores, n=k)


    return 0.0