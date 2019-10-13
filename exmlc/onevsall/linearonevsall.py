from __future__ import annotations

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

    def __init__(self,
                 clf: LinearClassifierMixin = SGDClassifier(loss='log'),
                 sparsify: bool = True,
                 n_jobs: int = 1,
                 verbose: bool = False):

        self.base_clf = clf
        self.sparsify = sparsify
        self.n_jobs = n_jobs
        self.verbose = verbose
        if not hasattr(self.base_clf, 'predict_proba'):
            delattr(self, 'predict_proba')


    def fit(self, X: Union[csr_matrix, np.ndarray], y:  Union[csr_matrix, np.ndarray]) -> LinearOneVsAllClf:
        # allocate memory for clf
        self.clf_store = np.empty(shape=(y.shape[1],), dtype='object')
        if self.verbose:
            print('Init clfs')
        for i in range(y.shape[1]):
            self.clf_store[i] = clone_estimator(self.base_clf)

        assert y.T.shape[0] == self.clf_store.shape[0]

        if self.n_jobs <= 1:
            for train_vector, clf in zip(y.T, self.clf_store):


        else:
            if self.verbose:
                print(f'Start fitting nodes with {self.n_jobs} workers')

            def fit_node(index_node, degree_tree, verbose):
                i, n = index_node
                n.fit_clf(X, n.y)
                if verbose:
                    print(f'Fitting node {i}/{degree_tree}')

            pool = Pool(self.n_jobs)
            pool.map(partial(fit_node, degree_tree=degree_tree, verbose=self.verbose),
                     enumerate(list(tree.bfs_traverse())))
        return tree

    def predict_proba(self, X, y) -> csr_matrix:
        return csr_matrix