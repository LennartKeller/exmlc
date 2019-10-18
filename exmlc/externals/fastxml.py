from __future__ import annotations

from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator


class FastXML(BaseEstimator):

    def __init__(self, fastxml_dir: str = 'FastXML'):
        pass

    def fit(self, X: csr_matrix, y: csr_matrix, **fit_params) -> FastXML:
        # 0. check if everything is installed
        # 0.1. create temp directories to store files
        # 1. convert and dump data to disk
        # 2. convert data into special clf format (using provided perl script)
        # 3. fit the model
        return self

    def decision_function(self, X: csr_matrix) -> csr_matrix:
        # 0. load the fitted model
        # 1. predict test data
        # 2. load predictions from disk into mem as csr_matrix
        return csr_matrix
