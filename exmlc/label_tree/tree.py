from __future__ import annotations

from collections import deque
from typing import *

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model.base import LinearClassifierMixin


class HuffmanNode:
    """
    Node of the (sort of) huffman tree used by the probabilistic label tree.
    It is used to store the node classifier and the indices to the instances used to train it.
    It also stores the probability (e.g. relative frequency in train data)
    of the tag or the joined probability of the tags it holds.
    Attributes like fitted or visited_while_prediction are only used for debugging purposes.
    It also holds the binary vector used for fitting the instance
    """
    def __init__(self,
                 probability: float,
                 label_idx: list,
                 children: List[HuffmanNode],
                 clf: LinearClassifierMixin):
        """
        Constructor for the node class
        :param probability: Relative Frequency of the tag or the tags in train data
        :param label_idx: Label index or indices of the labels of this the nodes instance
        :param children: list of all children of this instance
        :param clf: unfitted instance of the node classifier
        """
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
        Wrapper method for for fitting the node classifier at training.
        Even though it could be directly fitted by accessing the clf ttrribute this method
        contains exception handling for the case that the labels of the instance do not hold any train data.
        :param X: Features in sparse representation of shape (n_samples, n_features)
        :param y: binary vector of shape(n_samples,) indicating
        whether the sample is assigned to the tag or the tags of the instance or not
        :return: None
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
        """
        Predict the probability/probabilities of belonging to the tag or tags of the instances for each sample in X.
        :param X: Features in sparse representation of shape (n_samples, n_features)
        :return: vector of shape(n_samples) containing the probabilities for each sample in X
        """
        self.visited_while_prediction += 1
        return self.clf.predict_proba(X)

    def clf_decision_function(self, X: Union[np.ndarray, csr_matrix]) -> np.ndarray:
        """
        Predict the decision score(s) of belonging to the tag or tags of the instances for each sample in X.
        :param X: Features in sparse representation of shape (n_samples, n_features)
        :return: vector of shape(n_samples) containing the decision scores for each sample in X
        """
        self.visited_while_prediction += 1
        return self.clf.decision_function(X)

    def is_leaf(self) -> bool:
        """
        Whether or not the instance is a leave in the tree.
        :return: True if is leave else False
        """
        if not self.children:
            return True
        return False

    def get_children(self) -> List[HuffmanNode, HuffmanNode]:
        """
        Returns all children of the instance.
        :return: List of children as HuffmanNodes
        """
        return self.children

    def __gt__(self, huffmannode2: HuffmanNode) -> bool:
        """
        Implements comparison for the nodes.
        Returns True if the probability of the instance is greater than the probability of the other one.
        :param huffmannode2: other Node to check
        :return:
        """
        return self.probability > huffmannode2.probability

    def __eq__(self, huffmannode2: HuffmanNode) -> bool:
        """
        Implements comparison for the nodes.
        Returns True if the probability of the instance is equal to  the probability of the other one.
        :param huffmannode2: other Node to check
        :return:
        """
        return self.probability == huffmannode2.probability

    def __str__(self) -> str:
        """
        String representation of the instance. Used for debugging purposes.
        :return:
        """
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
