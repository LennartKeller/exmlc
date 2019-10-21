from __future__ import annotations

from collections import deque
from typing import *

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model.base import LinearClassifierMixin


class HuffmanNode:

    def __init__(self,
                 probability: float,
                 label_idx: list,
                 children: List[HuffmanNode],
                 clf: LinearClassifierMixin):

        self.train_idx = None
        self.y = None
        self.label_idx = label_idx
        self.probability = probability
        self.children = children
        self.fitted = False
        self.visited_while_prediction = 0
        if isinstance(clf, LinearClassifierMixin):
            self.clf = clf
        else:
            raise Exception('Node clf has to be a linear model.')

    def fit_clf(self, X: Union[np.ndarray, csr_matrix], y: Union[np.ndarray, csr_matrix]) -> None:
        """
        TODO
        :param X:
        :param y:
        :return:
        """
        try:
            self.clf.fit(X, y).sparsify()
            self.fitted = True
        except ValueError as e:
            # TODO
            raise Exception(f'Could not fit node {self}.\n'
                            f'This  either happens if there are labels in the dataset\n'
                            f'which are never assigned to a instance or assigned to all instances.\n'
                            f'Or while building the label tree labels were grouped\n'
                            f'together which are together assigned to all instances\n'
                            f'For the first case you could the prune_labels function to filter these labels.\n'
                            f'For second case there is currently no built in solution.\n'
                            f'Note: If you encounter this your dataset is very unlikely to at be at xmlc scale.\n')

    def clf_predict_proba(self, X: Union[np.ndarray, csr_matrix]) -> np.ndarray:
        self.visited_while_prediction += 1
        return self.clf.predict_proba(X)

    def clf_decision_function(self, X: Union[np.ndarray, csr_matrix]) -> np.ndarray:
        self.visited_while_prediction += 1
        return self.clf.decision_function(X)

    def is_leaf(self) -> bool:
        if not self.children:
            return True
        return False

    def get_children(self) -> List[HuffmanNode, HuffmanNode]:
        return self.children

    def __gt__(self, huffmannode2: HuffmanNode) -> bool:
        return self.probability > huffmannode2.probability

    def __eq__(self, huffmannode2: HuffmanNode) -> bool:
        return self.probability == huffmannode2.probability

    def __str__(self) -> str:
        if not self.is_leaf:
            return f"Prob: {self.probability}\nIs leaf: {self.is_leaf()}\n"
        return f"Label IDs: {self.label_idx}\nProb: {self.probability}\nIs leaf: {self.is_leaf()}\n"


class HuffmanTree:

    def __init__(self, root: HuffmanNode):
        self.root = root

    def dfs_traverse(self,
                     start_node: HuffmanNode = None,
                     yield_start_node: bool = False) -> Generator[HuffmanNode, None, None]:

        if not start_node:
            start_node = self.root
        if yield_start_node:
            yield start_node
        fifo = deque([])
        fifo.extendleft(start_node.get_children())
        while fifo:
            current_node = fifo.popleft()
            yield current_node
            if not current_node.is_leaf():
                fifo.extendleft(current_node.get_children())

    def bfs_traverse(self,
                     start_node: HuffmanNode = None,
                     yield_start_node: bool = False) -> Generator[HuffmanNode, None, None]:

        if not start_node:
            start_node = self.root
        if yield_start_node:
            yield start_node
        lifo = deque([])
        lifo.extendleft(start_node.get_children())
        while lifo:
            current_node = lifo.pop()
            yield current_node
            if not current_node.is_leaf():
                lifo.extendleft(current_node.get_children())

    def __len__(self) -> int:
        return len(list(self.bfs_traverse())) + 1  # plus root node
