from typing import *

from scipy.sparse import csr_matrix, lil_matrix
from sklearn.metrics import f1_score, label_ranking_average_precision_score

from .precision import sparse_average_precision_at_k
from .dcg import average_discounted_cumulative_gain_at_k


def extreme_multilabel_classification_report(y_true: csr_matrix,
                                             y_score: csr_matrix,
                                             k_range: Iterable = range(1, 11)) -> dict:
    """
    TODO docs
    1. Precision at k
    2. DCG at k
    3. nDCG at k
    4. F1 (macro) score
    :param y_true:
    :param y_score:
    :param k_range:
    :return:
    """
    # TODO use sklearn function to check dimensions
    if y_true.shape != y_score.shape:
        raise Exception('y_true and y_score must have same dimension')

    # init dict
    result = dict()
    result['precision@k'] = {}
    result['dcg@k'] = {}

    # precision at k
    for k in k_range:
        result['precision@k'][str(k)] = sparse_average_precision_at_k(y_true, y_score, k=k)
        result['dcg@k'][str(k)] = average_discounted_cumulative_gain_at_k(y_true, y_score, k=k)

    # TODO nDCG

    # F1 Macro Average
    # cast scores to binary matrix
    binary_pred = lil_matrix(y_score.shape, dtype='int8')
    binary_pred[y_score.nonzero()] = 1
    # binary_pred = binary_pred.tocsr()

    result['f1_marco'] = f1_score(y_true, binary_pred, average='macro')
    result['label_ranking_average_precision_score'] = label_ranking_average_precision_score(y_true, y_score)

    return result
