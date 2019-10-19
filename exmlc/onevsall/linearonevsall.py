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
from joblib import Parallel, delayed

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

        if self.n_jobs <= 1:  # sequential fitting
            for i in range(self.clf_store_.shape[0]):
                self.clf_store_[i].fit(X, y.T[i].toarray().ravel())
                if self.sparsify:
                    self.clf_store_[i].sparsify()

                if self.verbose:
                    print(f'Fitting clf {i + 1}/{self.clf_store_.shape[0]}')
        else:  # parallel fitting
            if self.verbose:
                print(f'Start fitting nodes with {self.n_jobs} workers')

            def fit_clf(index: int,
                        X: Union[csr_matrix, np.ndarray],
                        y: csr_matrix,
                        clf_store: np.ndarray,
                        sparsify: bool,
                        verbose: bool) -> None:
                """
                Map function for pool multiprocessing
                :param index: index of clf to process
                :param X: train data
                :param y: binary label vec for one distinct label
                :param clf_store: array with label clfs
                :param sparsify: see above
                :param verbose: see above
                :return: None
                """

                train_vec = y.T[index].toarray().ravel().copy()

                clf_store[index].fit(X, train_vec)

                if sparsify:
                    clf_store[index].sparsify()
                if verbose:
                    print(f'Fitting clf {index + 1}/{clf_store.shape[0]}')
                return clf_store[index]

            # pool = Pool(self.n_jobs)
            # pool.map(
            #     partial(fit_clf,
            #             X=X.copy(), y=y.copy(),
            #             sparsify=self.sparsify,
            #             clf_store=self.clf_store_,
            #             verbose=self.verbose),
            #     #Array('i', list(range(self.clf_store_.shape[0])))
            #     range(self.clf_store_.shape[0])
            # )

            parallel = Parallel(self.n_jobs)
            results = parallel(
                delayed(fit_clf)(clf_index, X, y, self.clf_store_, self.sparsify, self.verbose) for clf_index in list(range(self.clf_store_.shape[0]))
            )
            self.clf_store_ = np.array(results)
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
            if self.verbose:
                print(f'Predicting class for label {clf_index}')
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
            if self.verbose:
                print(f'Predicting probability for label {clf_index}')
            label_probabilities = self.clf_store_[clf_index].predict_proba(X)

            # filter probs because we're only interested in the prob of the label
            y_pred.append([i[1] for i in label_probabilities])

        print(y_pred)
        return np.array(y_pred).T

    def decision_function(self, X: Union[csr_matrix, np.ndarray]) -> np.ndarray:
        if not hasattr(self, 'clf_store_'):
            raise NotFittedError
        y_scores = list()
        for clf_index in range(self.clf_store_.shape[0]):
            if self.verbose:
                print(f'Predicting decision score for label {clf_index}')
            label_scores = self.clf_store_[clf_index].decision_function(X)
            y_scores.append(label_scores)
        return np.array(y_scores).T


if __name__ == '__main__':
    # Just for debugging

    from sklearn.datasets import make_multilabel_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    from sklearn.linear_model import LogisticRegression
    from exmlc.metrics import sparse_average_precision_at_k
    from sklearn.svm import LinearSVC

    X, y = make_multilabel_classification(n_classes=100,
                                          n_features=1000,
                                          n_samples=10000,
                                          sparse=True,
                                          return_indicator='sparse',
                                          allow_unlabeled=False,
                                          random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    custom_sgd = LogisticRegression()

    print('Fitting')
    clf = OneVsAllLinearClf(n_jobs=-1, verbose=True)
    clf.fit(X_train, y_train)

    print('Predicting')
    y_pred = clf.predict(X_test)

    print(classification_report(y_test, y_pred))

    print('#' * 100)
    print('Using nonlinear clf')

    clfnl = OneVsAllLinearClf(clf=LinearSVC(), n_jobs=-1, sparsify=False, verbose=True)
    clfnl.fit(X_train, y_train)
    y_prednl = clfnl.decision_function(X_test)

    print(sparse_average_precision_at_k(y_test, csr_matrix(y_prednl)))
