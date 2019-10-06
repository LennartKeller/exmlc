from typing import *
from scipy.sparse import csr_matrix


def average_discounted_cumulative_gain_at_k(y_true: csr_matrix,
                                            y_scores: csr_matrix,
                                            normalize: bool = False) -> float:
    if normalize:
        raise NotImplementedError

    return 0.0