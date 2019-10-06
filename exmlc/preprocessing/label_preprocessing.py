import warnings
from typing import *

from scipy.sparse import csr_matrix
from sklearn.base import TransformerMixin
from sklearn.exceptions import NotFittedError


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

    if y_train.shape[1] != y_test.shape[1]:
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


class MultiLabelIndexer(TransformerMixin):

    def fit(self, X, y=None):
        label_set = set()
        for entry in X:
            for label in entry:
                label_set.add(label)

        self.classes_ = list(sorted(label_set))
        self._look_up_table = dict(zip(self.classes_, range(len(self.classes_))))

        return self

    def transform(self, X, y=None):
        if not hasattr(self, 'classes_'):
            raise NotFittedError

        X_t = []
        self.unknown_labels_ = set()

        for entry in X:
            x_t = []

            for label in entry:
                try:
                    x_t.append(self._look_up_table[label])
                except KeyError:
                    self.unknown_labels_.add(label)

            X_t.append(x_t)

        if self.unknown_labels_:
            warnings.warn(
                f'Ignored labels {list(sorted(self.unknown_labels_))} because they do not occur in train vocab.',
                UserWarning
            )

        return X_t

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X, y=None):
        X_inv = []
        for entry in X:
            x_inv = []
            for ind in entry:
                x_inv.append(self.classes_[ind])
            X_inv.append(x_inv)
        return X_inv
