import numpy as np
from scipy.sparse import csr_matrix

from .utils import get_n_top_values_sorted


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
        dcg = 0
        for rank, ind in enumerate(idx, 1):
            if true_row[0, ind] > 0:
                dcg += np.divide(1, np.log(rank + 1))
        values.append(np.divide(dcg, k))

    return np.mean(values)  # , values


def discounted_culmulative_gain_at_k(y_true: csr_matrix, y_scores: csr_matrix, k: int = 3,
                                     normalize: bool = False) -> float:
    if normalize:
        raise NotImplementedError

    if y_true.shape != y_scores.shape:
        raise Exception('y_true and y_pred must have same shape')
    if y_true.shape[0] > 1:
        raise Exception

    k_top_idx = get_n_top_values_sorted(y_scores, n=k)[0]

    dcg = 0
    for rank, ind in enumerate(k_top_idx, 1):
        if y_true[0, ind] > 0:
            dcg += np.divide(1, np.log(rank + 1))  # y_true[0, ind] always equals one
    return np.divide(dcg, k)


def avg_dcg_at_k(y_true: csr_matrix, y_scores: csr_matrix, k: int = 3, normalize: bool = False) -> float:
    if normalize:
        raise NotImplementedError

    if y_true.shape != y_scores.shape:
        raise Exception('y_true and y_pred must have same shape')
    if y_true.shape[1] < k:
        raise Exception('Less labels than k')
    results = []
    for true_vec, score_vec in zip(y_true, y_scores):
        results.append(discounted_culmulative_gain_at_k(true_vec, score_vec, k=k, normalize=normalize))
    return np.mean(results)
