from typing import *

from scipy.sparse import csr_matrix


def prune_labels(y_train: Union[csr_matrix],
                 y_test: Union[csr_matrix],
                 threshold=1,
                 verbose=False) -> Tuple[csr_matrix, csr_matrix, list]:
    """
    Deletes all labels in train and test set which occur less often then the threshold in the train set.
    :param y_train:
    :param y_test:
    :param threshold:
    :param verbose:
    :return:
    """

    if X.shape[0] != y.shape[0]:
        raise Exception('X and y must have same shape')


    y_train = y_train.tolil()
    y_test = y_test.tolil()

    label_idx_to_remove = []
    for index, label_vector in enumerate(y_train.T):
        if label_vector.count_nonzero() < threshold:
            label_idx_to_remove.append(index)

    total_idx = list(range(y_train.T.shape[0]))
    pruned_idx = list(set(total_idx) - set(label_idx_to_remove))

    # remove labels
    y_train = y_train.T[pruned_idx].T
    y_test = y_test.T[pruned_idx].T

    if verbose:
        print(f'Dropped {len(label_idx_to_remove)} labels because they occur less than {threshold} times.')

    return y_train.tocsr(), y_test.tocsr(), label_idx_to_remove
