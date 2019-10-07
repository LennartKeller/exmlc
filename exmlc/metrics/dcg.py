from typing import *
from scipy.sparse import csr_matrix
from .utils import get_n_top_values_sorted
import numpy as np


def average_discounted_cumulative_gain_at_k(y_true: csr_matrix,
                                            y_scores: csr_matrix,
                                            k: int = 3,
                                            normalize: bool = False) -> float:  # -> Tuple[np.ndarray, List[float]]:
    """
    TODO
    :param y_true:
    :param y_scores:
    :param k:
    :param normalize:
    :return:
    """

    if normalize:
        # TODO
        raise NotImplementedError

    if y_true.shape != y_scores.shape:
        raise Exception('y_true and y_pred must have same shape')
    if y_true.shape[1] < k:
        raise Exception('Less labels than k')

    k_top_idx = get_n_top_values_sorted(y_scores, n=k)

    values = []
    for true_row, idx in zip(y_true, k_top_idx):
        for rank, ind in enumerate(idx, 1):
            if true_row[0, ind] > 0:
                values.append(np.divide(1, np.log(rank + 1)))

    return np.mean(values)  # , values


