from __future__ import annotations

from collections import deque
from heapq import heappop, heapify, heappush
from itertools import chain, repeat
from multiprocessing.dummy import Pool
from os import cpu_count
from typing import *

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator
from sklearn.base import clone as clone_estimator
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model.base import LinearClassifierMixin
from tqdm import tqdm

from .tree import HuffmanNode, HuffmanTree
from ..metrics import sparse_average_precision_at_k


class PLTClassifier(BaseEstimator):
    """
    Implementation of the Probabilistic Label Tree
    """

    def __init__(self,
                 threshold: float = .005,
                 node_clf: LinearClassifierMixin = SGDClassifier(alpha=1e-05,
                                                                 eta0=0.1,
                                                                 learning_rate='constant',
                                                                 loss='modified_huber',
                                                                 penalty='l1',
                                                                 random_state=42),
                 num_children: int = 2,
                 use_probs: bool = True,
                 n_jobs: int = 1,
                 verbose: bool = True) -> None:

        self.threshold = threshold
        self.node_clf = node_clf
        self.num_children = num_children
        self.use_probs = use_probs

        if n_jobs == -1:
            self.n_jobs = cpu_count()
        else:
            self.n_jobs = n_jobs

        self.verbose = verbose

    def fit(self, X: Union[np.ndarray, csr_matrix], y: Union[np.ndarray, csr_matrix]) -> PLTClassifier:
        """
        Fits the classifier
        :param X: sparse feature matrix
        :param y: sparse binary label matrix
        :return:
        """

        self.yi_shape_ = y[0].shape
        self.tree_ = self._create_huffman_tree(y, k=self.num_children)
        self.tree_ = self._assign_train_indices(self.tree_, y)
        self.tree_ = self._fit_tree(self.tree_, X)

        return self

    def predict(self, X: Union[np.ndarray, csr_matrix], use_probs: bool = None) -> csr_matrix:
        """
        predicts labels
        :param X: sparse feature matrix
        :param use_probs: experimental arg do not modify it
        :return:
        """

        if not self.tree_:
            raise NotFittedError

        if use_probs is None:
            use_probs = self.use_probs

        y_pred = []

        if self.verbose:
            data_iterator = tqdm(enumerate(X), total=X.shape[0])
        else:
            data_iterator = enumerate(X)

        if self.verbose:
            print('Predicting samples')

        for index, x in data_iterator:
            # if self.verbose:
            #     print(f'Predicting sample {index + 1}/{X_length}')
            y_pred.append(self._traverse_tree_prediction(self.tree_, x, use_probs))

        return csr_matrix(y_pred)

    def decision_function(self, X: Union[np.ndarray, csr_matrix], use_probs: bool = None) -> csr_matrix:
        """
        Returns probabilities of the selected labels
        :param X: feature matrix with samples to predict
        :return: label matrix with probabilities instead of binary indicators.
        """

        if not hasattr(self, 'tree_'):
            raise NotFittedError

        if use_probs is None:
            use_probs = self.use_probs

        y_pred_decision = []

        if self.verbose:
            data_iterator = tqdm(enumerate(X), total=X.shape[0])
        else:
            data_iterator = enumerate(X)

        if self.verbose:
            print('Predicting decision scores')

        for index, x in data_iterator:
            y_pred_decision.append(self._traverse_tree_decision_function(self.tree_, x, use_probs))
        return csr_matrix(y_pred_decision)

    def score(self, X_test: csr_matrix, y_test: csr_matrix, k: int = 3) -> float:
        """
        predicts label scores for X_test and computes scores
        :param X_test: feature matrix
        :param y_test: ground truth label matrix
        :param k: k value for average precision@k
        :return:
        """

        if not self.tree_:
            raise NotFittedError

        y_scores = self.decision_function(X_test)

        return sparse_average_precision_at_k(y_test, y_scores, k=k)

    def _create_huffman_tree(self, y: csr_matrix, k: int) -> HuffmanTree:
        """
        Create a huffman tree_ based on the given labels and their probabilities.
        :param y: sparse binary representation label vectors
        :return: huffman label tree_
        """

        if self.verbose:
            print('Building tree_')

        label_probs = self._compute_label_probabilities(y)

        priority_queue = []
        for label_id, prob in label_probs.items():
            new_node = HuffmanNode(probability=prob,
                                   clf=clone_estimator(self.node_clf),
                                   label_idx=[label_id],
                                   children=[])
            priority_queue.append(new_node)
        heapify(priority_queue)

        while len(priority_queue) > 1:
            n_children = [heappop(priority_queue) for _ in range(min(k, len(priority_queue)))]
            new_node = HuffmanNode(probability=sum(map(lambda node: node.probability, n_children)),
                                   clf=clone_estimator(self.node_clf),
                                   label_idx=list(chain.from_iterable(map(lambda node: node.label_idx, n_children))),
                                   children=n_children)

            heappush(priority_queue, new_node)

        if len(priority_queue) != 1:
            raise Exception('Building tree failed (Priority Queue must only contain one element)')

        return HuffmanTree(root=priority_queue[0])

    def _assign_train_indices(self, tree: HuffmanTree, y: csr_matrix) -> HuffmanTree:
        """
        Assings indices of train data to the nodes of the label tree.
        :param tree: Label Tree
        :param y: Label Matrix
        :return: tree with data assigned
        """

        if self.verbose:
            print('Assigning train data to tree_ nodes')

        for node in tree.bfs_traverse():
            compare_row = np.zeros(y[0].shape).ravel()
            compare_row[node.label_idx] = 1
            compare_row = csr_matrix(compare_row)
            # do matrix vector multiplication to
            node.train_idx = y.multiply(compare_row)
            node.y = node.train_idx.max(axis=1).astype('int8').toarray().ravel()

        return tree

    def _fit_tree(self, tree: HuffmanTree, X: np.ndarray) -> HuffmanTree:
        """
        Fits the tree.
        The tree is traversed in a breath-first style and the classifier at each not is fitted
        with the binary y_train data which is assigned to that node.
        :param tree: huffman label tree_.
        :param X: Features to train the classifiers
        :return: the trained huffman label tree_
        """

        degree_tree = len(tree)

        if self.n_jobs <= 1:
            for index, node in enumerate(list(tree.bfs_traverse())):
                if self.verbose:
                    print(f'Fitting node {index + 1}/{degree_tree - 1}')  # minus root node which is not fitted
                node.fit_clf(X, node.y)
        else:
            if self.verbose:
                print(f'Start fitting nodes with {self.n_jobs} workers')

            def fit_node(index_node):
                i, n = index_node
                n.fit_clf(X, n.y)

            pool = Pool(self.n_jobs)
            if self.verbose:
                for _ in tqdm(pool.imap(fit_node, enumerate(list(tree.bfs_traverse()))), total=len(self.tree_) - 1):
                    pass
            else:
                pool.map(fit_node, enumerate(list(tree.bfs_traverse())))

            pool.close()

        return tree

    def _traverse_tree_prediction(self, tree: HuffmanTree, x: np.ndarray, use_probs: bool) -> np.ndarray:
        """
        Traverses the given label tree_ with a single feature instance to predict (in a depth-first manner).
        At each node the decision of whether continuing to the nodes children or this subtree is made.
        Therefore the nodes classifiers prediction is used as well as the prediction of the previous node.
        If leaf nodes are reached the corresponding labels are returned.
        :param tree: huffman label tree
        :param x: single instance of the data to predict
        :param use_probs:
        :return: numpy array containing the label indices.
        """

        yi_pred = []
        fifo = deque([])
        prev_prob = 1.0
        fifo.extendleft(((children, pr_prob) for children, pr_prob in zip(tree.root.get_children(), repeat(prev_prob))))
        while fifo:
            current_node, prev_prob = fifo.popleft()

            if use_probs:
                prob = current_node.clf_predict_proba(x).ravel()[1].item()
            else:
                prob = current_node.clf_decision_function(x).item()

            if prob * prev_prob < self.threshold:
                continue

            # prev_prob = prob
            new_prev_prob = prob * prev_prob

            if not current_node.is_leaf():
                fifo.extendleft(
                    (
                        (children, pr) for children, pr in zip(current_node.get_children(), repeat(new_prev_prob))
                    )
                )

            if current_node.is_leaf():
                if len(current_node.label_idx) != 1:
                    raise Exception('Leaf node has more than one label associated.')
                yi_pred.append(current_node.label_idx[0])

        yi_vector = np.zeros(self.yi_shape_)
        yi_vector[0, yi_pred] = 1

        return yi_vector.astype('int8').ravel()

    def _traverse_tree_decision_function(self, tree: HuffmanTree, x: np.ndarray, use_probs: bool) -> np.ndarray:
        """
        Experimental do not use it
        """

        yi_pred = []
        yi_prob = []
        fifo = deque([])
        prev_prob = 1.0
        fifo.extendleft(((children, pr_prob) for children, pr_prob in zip(tree.root.get_children(), repeat(prev_prob))))

        while fifo:
            current_node, prev_prob = fifo.popleft()
            if use_probs:
                prob = current_node.clf_predict_proba(x).ravel()[1].item()
            else:
                prob = current_node.clf_decision_function(x).item()  # use continuous values

            if prob * prev_prob < self.threshold:
                continue

            new_prev_prob = prob * prev_prob
            if not current_node.is_leaf():
                fifo.extendleft(
                    (
                        (children, pr) for children, pr in zip(current_node.get_children(), repeat(new_prev_prob))
                    )
                )

            if current_node.is_leaf():
                assert len(current_node.label_idx) == 1, Exception('Leaf node has more than one label associated.')
                yi_pred.append(current_node.label_idx[0])
                yi_prob.append(prob)

        yi_vector = np.zeros(self.yi_shape_, dtype='float64')
        yi_vector[0, yi_pred] = yi_prob
        return yi_vector.ravel()

    def _compute_label_probabilities(self, y: Union[np.ndarray, csr_matrix]) -> Dict[int, float]:
        """
        Computes the label probabilities e.g. their relative frequencies.
        :param y: label in binary representation
        :return: dict with the label indices as keys and their probabilities as values
        """
        label_probs = {}
        total_no_tags = np.nonzero(y)[0].shape[0]
        for index, column in enumerate(y.T):
            counter = np.nonzero(column)[0].shape[0]
            label_probs[index] = counter / total_no_tags
        return label_probs


if __name__ == '__main__':
    from sklearn.datasets import make_multilabel_classification
    from sklearn.model_selection import train_test_split
    from exmlc.metrics import sparse_average_precision_at_k
    from sklearn.metrics import classification_report

    X, y = make_multilabel_classification(n_samples=100,
                                          n_classes=10,
                                          n_features=500,
                                          allow_unlabeled=False,
                                          sparse=True,
                                          return_indicator='sparse',
                                          random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)

    plt = PLTClassifier(n_jobs=-1, verbose=True, use_probs=False)

    print('Start training')
    plt.fit(X_train, y_train)

    print('Start predicting')
    scores = plt.decision_function(X_test)

    print(sparse_average_precision_at_k(y_test, scores, k=1))

    predictions = plt.predict(X_test)
    print(classification_report(y_test, predictions))
