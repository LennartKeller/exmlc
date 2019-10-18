from __future__ import annotations

from functools import partial
from multiprocessing.dummy import Pool, Array
from os import cpu_count
from typing import *

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, issparse
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model.base import LinearClassifierMixin
from sklearn.utils import check_X_y


class OneVsAllLinearClf(BaseEstimator):
    """
    TODO
    """

    def __init__(self,
                 clf: LinearClassifierMixin = SGDClassifier(loss='hinge'),
                 sparsify: bool = True,
                 n_jobs: int = 1,
                 verbose: bool = False):

        self.base_clf = clf
        self.sparsify = sparsify

        if n_jobs == -1:
            self.n_jobs = cpu_count()
        else:
            self.n_jobs = n_jobs

        self.verbose = verbose

    def fit(self, X: Union[csr_matrix, np.ndarray], y: csr_matrix) -> OneVsAllLinearClf:
        """
        TODO
        :param X:
        :param y:
        :return:
        """
        check_X_y(X, y,
                  accept_sparse=True,
                  accept_large_sparse=True,
                  multi_output=True)
        if not issparse(y):
            raise Exception('y has to be sparse')

        if self.verbose:
            print('Init classifiers')

        # allocate memory for clfs
        self.clf_store_ = np.full((y.shape[1],), self.base_clf, dtype=np.dtype(LinearClassifierMixin))

        if self.n_jobs <= 1:
            for i in range(self.clf_store_.shape[0]):
                self.clf_store_[i].fit(X, y.T[i].toarray().ravel())
                if self.sparsify:
                    self.clf_store_[i].sparsify()

                if self.verbose:
                    print(f'Fitting clf {i + 1}/{self.clf_store_.shape[0]}')
        else:
            if self.verbose:
                print(f'Start fitting nodes with {self.n_jobs} workers')

            def fit_clf(index: int,
                        X: Union[csr_matrix, np.ndarray],
                        y: csr_matrix,
                        clf_store: np.ndarray,
                        sparsify: bool,
                        verbose: bool):

                clf_store[index].fit(X, y.T[index].toarray().ravel())
                if sparsify:
                    clf_store[index].sparsify()
                if verbose:
                    print(f'Fitting clf {index + 1}/{clf_store.shape[0]}')

                return

            pool = Pool(self.n_jobs)
            pool.map(
                partial(fit_clf,
                        X=X, y=y,
                        sparsify=self.sparsify,
                        clf_store=self.clf_store_,
                        verbose=self.verbose),
                Array('i', range(self.clf_store_.shape[0]))
            )

        return self

    def predict(self, X: Union[np.ndarray, csr_matrix]) -> csr_matrix:

        if not hasattr(self, 'clf_store_'):
            raise NotFittedError

        # sequential prediction (n_jobs==1)
        # while prediction each label clf will predict if the label should be given
        # to each sample in X
        # so we init a sparse matrix with transposed structure (n_label, n_samples)
        # to easy insert the results
        y_pred_transposed = lil_matrix(
            (self.clf_store_.shape[0], X.shape[0]),
            dtype='int8'
        )

        for clf_index in range(self.clf_store_.shape[0]):
            label_predictions = self.clf_store_[clf_index].predict(X)
            y_pred_transposed[clf_index, np.nonzero(label_predictions)] = 1
        return y_pred_transposed.T.tocsr()

    def predict_proba(self, X: Union[np.ndarray, csr_matrix]) -> np.ndarray:

        if not hasattr(self.clf_store_[0], 'predict_proba'):
            raise Exception('Base classifier does not support probability predictions')

        if not hasattr(self, 'clf_store_'):
            raise NotFittedError
        # sequential prediction (n_jobs==1)
        y_pred = list()
        for clf_index in range(self.clf_store_.shape[0]):
            label_probabilities = self.clf_store_[clf_index].predict_proba(X)
            y_pred.append([i[1] for i in label_probabilities])

        print(y_pred)
        return np.array(y_pred).T


if __name__ == '__main__':
    # Just for debugging

    from sklearn.datasets import make_multilabel_classification
    from sklearn.model_selection import train_test_split
    from exmlc.metrics import sparse_average_precision_at_k

    X, y = make_multilabel_classification(sparse=True, return_indicator='sparse')
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=31)

    custom_sgd = SGDClassifier(loss='modified_huber')


    print('Fitting')
    clf = OneVsAllLinearClf(clf=custom_sgd, n_jobs=1, verbose=True)
    clf.fit(X_train, y_train)

    print('Predicting')
    y_pred = clf.predict_proba(X_test)

    print(sparse_average_precision_at_k(y_test, csr_matrix(y_pred)))
