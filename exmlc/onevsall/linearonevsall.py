from __future__ import annotations
from os import cpu_count
from typing import *
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.base import BaseEstimator
from sklearn.base import clone as clone_estimator
from sklearn.linear_model.base import LinearClassifierMixin
from sklearn.linear_model import SGDClassifier
import numpy as np
from multiprocessing.dummy import Pool
from functools import partial


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
        if not hasattr(self.base_clf, 'predict_proba'):
            delattr(self, 'predict_proba')

    def fit(self, X: Union[csr_matrix, np.ndarray], y:  Union[csr_matrix, np.ndarray]) -> OneVsAllLinearClf:
        """
        TODO
        :param X:
        :param y:
        :return:
        """
        # allocate memory for clfs

        if self.verbose:
            print('Init classifiers')

        self.clf_store = np.full((y.shape[1],), self.base_clf, dtype='object')

        assert y.T.shape[0] == self.clf_store.shape[0]

        if self.n_jobs <= 1:
            for i in range(self.clf_store.shape[0]):
                self.clf_store[i].fit(X, y.T[i])
                if self.sparsify:
                    self.clf_store[i].sparsify()

                if self.verbose:
                    print(f'Fitting clf {i + 1}/{self.clf_store.shape[0]}')
        else:
            if self.verbose:
                print(f'Start fitting nodes with {self.n_jobs} workers')

            def fit_clf(index, clf_store, verbose, sparsify):
                clf_store[index].fit_clf(X, y.T[i])
                if sparsify:
                    clf_store[index].sparsify()
                if verbose:
                    print(f'Fitting clf {index}/{clf_store.shape[0]}')

            pool = Pool(self.n_jobs)
            pool.map(partial(fit_clf,
                             X=X, y=y,
                             sparsify=self.sparsify,
                             clf_store=self.clf_store,
                             verbose=self.verbose),
                     enumerate(self.clf_store))

        return self

    def predict_proba(self, X, y) -> csr_matrix:
        return csr_matrix