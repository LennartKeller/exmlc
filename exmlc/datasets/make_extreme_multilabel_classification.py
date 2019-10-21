from typing import *

from numpy import array
from scipy.sparse import csr_matrix
from sklearn.datasets import make_multilabel_classification


def make_extreme_multilabel_classification(n_samples: int = 10000,
                                           n_features: int = 5000,
                                           n_classes: int = 100,
                                           n_labels: int = 5,
                                           allow_unlabeled: bool = False,
                                           return_distributions: bool = False,
                                           random_state: int = None) -> Tuple[csr_matrix, csr_matrix,
                                                                       Optional[array], Optional[array]]:
    """
    Simple wrapper for sklearns make_multilabel_classification
    function with parameters suitable for extreme multilabel classification.
    NOTE: Unfortunately the probability distribution of the returned dataset isn't long tailed
    like in most extreme multilabel cases (e.g. there are few samples which are only assigned few times)
    For more information visit
    https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_multilabel_classification.html
    TODO Think about a way to do this.
    :param n_samples: number of samples | default 10000
    :param n_features: number of features | default 50
    :param n_classes: number of total labels | default 100
    :param n_labels: number of labels per instance | default 5
    :param allow_unlabeled: whether or not the dataset contains unlabeled samples | default False
    :param return_distributions: whether or not the distribution should be returned | default False
    :param random_state: random_state for creating data | default None
    :return: X as csr_matrix, y as csr_matrix
    (p_c as array (probability of each class being drawn),
    p_w_c as array (probability of each feature being drawn given each class.))
    """
    return make_multilabel_classification(sparse=True, return_indicator='sparse', **locals())
