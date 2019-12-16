from __future__ import annotations

from os import cpu_count
from typing import *

import numpy as np
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix, lil_matrix, issparse
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model.base import LinearClassifierMixin
from sklearn.utils import check_X_y
from tqdm import tqdm


class OneVsAllLinearClf(BaseEstimator):
    """
    This class implements one of the most basic problem transformation approaches
    for multilabel classification tasks.
    The one vs all method trains a classifier for each tag in the train set
    to make a binary decision of whether given sample belongs to the tag or not.
    At prediction each classifier is evaluated to obtain the tags.
    While this method performs very good in most cases it scales with the number of labels
    and thus is often computationally infeasible on datasets of extreme multilabel classification scale.
    This problem not only concerns the memory usage but also the time required for training and often more important
    the prediction time since for each instance the whole set of classifiers has to be
    evaluated.

    The model itself does not have any parameters for fine-tuning,
    but the classifier used for each tag should be tuned carefully in order to obtain best results.
    If it becomes striking that the performance is harmed because too many tags are assigned to each instance
    a custom threshold could be used to decrease the number of tags. In this case a solution
    for non-probabilistic classifiers such as support vector machines has to be found.

    In order to overcome or least weaken the constraint of memory usage of the one vs all approach
    this implementation makes use of scikit-learns ability to convert the coefficient matrices
    of fitted linear models from dense to sparse format in order to decrease the memory usage.

    Also while training the classifiers multi-threading is used in order to speed up this process.

    Obviously the limitation on linear classification models restricts the theoretical performance
    of this approach. But in the case of text classification which is the most probable use cases of exmlc
    linear methods such as a support vector machine with a linear kernel are empirically proven
    to perform quiet well.

    To even further increase memory usage consider using L1-Regularization for the tag-classifiers since
    this produces even more sparse coefficient matrices. But keep in mind that this could also harm
    the performance of the model since L2 or Elasticnet often yield higher accuracy.

    Thus as stated above this problem transformation approach yield good results in most cases
    it also theoretically suffers from a problem arising from its design: The label imbalance.
    In most multilabel and nearly all exmlc datasets the labels are not equally distributed
    but in a very heavy tailed manner.
    This means that there is small of number of labels which occur very often while
    the majority of labels is only used very rarely (often not more than once or twice).
    Hence learning to assign the rare labels correctly is challenging.
    Even though this is a general problem shared by all extreme multilabel classification approaches
    in this case it is reinforced by the fact that by dividing the data
    into a set of binary classification decisions the imbalance is increased since even the most common
    tag is only assigned to a tiny minority of instances compared to the whole set of instances.

    Please note that despite of the type annotations you can also can use nonlinear classifiers
    if you set the sparsify parameter to False.
    """

    def __init__(self,
                 clf: LinearClassifierMixin = SGDClassifier(loss='hinge'),
                 sparsify: bool = True,
                 n_jobs: int = 1,
                 verbose: bool = False):
        """
        Constructor of the OneVsAllLinearClf class
        :param clf: base classifier used for each tag
        Defaults to a linear support vector machine which is trained using stochastic gradient descent.
        :param sparsify: whether or not the coefficient should be converted to sparse formatted after fitting
        :param n_jobs: Number of cores to use while training each clf
        :param verbose: Whether or not to print information while training and prediction
        """
        self.base_clf = clf
        self.sparsify = sparsify

        if n_jobs < 1:
            self.n_jobs = cpu_count()
        else:
            self.n_jobs = n_jobs

        self.verbose = verbose

    def fit(self, X: Union[csr_matrix, np.ndarray], y: csr_matrix) -> OneVsAllLinearClf:
        """
        Fits the model.
        For each tag in the training data a binary classifier model is trained.
        If sparsify is True the coefficient matrix of a trained instance will be formatted to sparse format
        which decreases the memory usage significantly.
        :param X: Features in sparse representation of shape (n_samples, n_features)
        :param y: sparse binary label matrix of shape (n_samples, n_labels)
        :return: fitted instance of itself
        """
        check_X_y(X, y,
                  accept_sparse=True,
                  accept_large_sparse=True,
                  multi_output=True)
        if not issparse(y):
            raise Exception('y has to be sparse')

        if self.verbose:
            print(f'Init {y.shape[1]}classifiers')

        # allocate memory for clfs
        self.clf_store_ = np.full((y.shape[1],), self.base_clf, dtype=np.dtype(LinearClassifierMixin))

        if self.n_jobs <= 1:  # sequential fitting
            for i in tqdm(range(self.clf_store_.shape[0])):
                self.clf_store_[i].fit(X, y.T[i].toarray().ravel())
                if self.sparsify:
                    self.clf_store_[i].sparsify()

                # if self.verbose:
                #     print(f'Fitting clf {i + 1}/{self.clf_store_.shape[0]}')
        else:  # parallel fitting
            if self.verbose:
                print(f'Start fitting with {self.n_jobs} workers')

            def fit_clf(index: int,
                        X: Union[csr_matrix, np.ndarray],
                        y: csr_matrix,
                        clf_store: np.ndarray,
                        sparsify: bool) -> None:
                """
                Map function for pool multiprocessing
                :param index: index of clf to process
                :param X: train data
                :param y: binary label vec for one distinct label
                :param clf_store: array with label clfs
                :param sparsify: see in constructor
                :param verbose: see parent method
                :return: None
                """

                train_vec = y.T[index].toarray().ravel().copy()

                clf_store[index].fit(X, train_vec)

                if sparsify:
                    clf_store[index].sparsify()
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
            if self.verbose:
                results = parallel(
                    delayed(fit_clf)(clf_index,
                                     X,
                                     y,
                                     self.clf_store_,
                                     self.sparsify) for clf_index in tqdm(list(range(self.clf_store_.shape[0])))
                )
            else:
                results = parallel(
                    delayed(fit_clf)(clf_index,
                                     X,
                                     y,
                                     self.clf_store_,
                                     self.sparsify) for clf_index in list(range(self.clf_store_.shape[0]))
                )

            self.clf_store_ = np.array(results)
        return self

    def predict(self, X: Union[np.ndarray, csr_matrix]) -> csr_matrix:
        """
        Predicts tags for each sample in X.
        For prediction the set of classifiers is traversed and each classifier
        takes a decision of whether the instance belongs to its tag or not.
        :param X: sparse feature representation of shape(n_samples, n_features)
        :return: binary label matrix in sparse format
        """
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

        if self.verbose:
            label_iterator = tqdm(range(self.clf_store_.shape[0]))
            print('Predicting labels')
        else:
            label_iterator = range(self.clf_store_.shape[0])

        for clf_index in label_iterator:
            label_predictions = self.clf_store_[clf_index].predict(X)
            y_pred_transposed[clf_index, np.nonzero(label_predictions)] = 1
        return y_pred_transposed.T.tocsr()

    def predict_proba(self, X: Union[np.ndarray, csr_matrix]) -> np.ndarray:
        """
        Acts mostly similar to the predict function but returns a label matrix
        containing the probability of each tag containing to the sample.
        Note this method is only available if the base classifier provides a predict_proba method.
        :param X: sparse feature representation of shape(n_samples, n_features)
        :return: label matrix with probabilities instead of binary indicators
        """
        if not hasattr(self.clf_store_[0], 'predict_proba'):
            raise Exception('Base classifier does not support probability predictions')

        if not hasattr(self, 'clf_store_'):
            raise NotFittedError
        # sequential prediction (n_jobs==1)

        y_pred = list()

        if self.verbose:
            label_iterator = tqdm(range(self.clf_store_.shape[0]))
            print('Predicting labels probabilities')
        else:
            label_iterator = range(self.clf_store_.shape[0])
        for clf_index in label_iterator:
            label_probabilities = self.clf_store_[clf_index].predict_proba(X)

            # filter probs because we're only interested in the prob of the label
            y_pred.append([i[1] for i in label_probabilities])

        return np.array(y_pred).T

    def decision_function(self, X: Union[csr_matrix, np.ndarray]) -> np.ndarray:
        """
        Acts mostly similar to the predict function but returns a label matrix
        containing the decision value of each tag containing to the sample.
        Note this method is only available if the base classifier provides a decision_function method.
        :param X: sparse feature representation of shape(n_samples, n_features)
        :return: label matrix with decision values instead of binary indicators
        """
        if not hasattr(self, 'clf_store_'):
            raise NotFittedError

        y_scores = list()

        if self.verbose:
            label_iterator = tqdm(range(self.clf_store_.shape[0]))
            print('Predicting label scores')
        else:
            label_iterator = range(self.clf_store_.shape[0])

        for clf_index in label_iterator:
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
