from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

from typing import *
from tqdm import tqdm

from sklearn.exceptions import NotFittedError
from sklearn.base import BaseEstimator
from flair.data import Sentence
from flair.embeddings import Embeddings, DocumentPoolEmbeddings, WordEmbeddings
from sklearn.neighbors import NearestNeighbors


class FlairEmbeddingsClassifier(BaseEstimator):

    def __init__(self,
                 word_embeddings: List[Embeddings] = (WordEmbeddings('de'), WordEmbeddings('de-crawl')),
                 pooling: str = 'mean',
                 fine_tune_mode: str = 'nonlinear',
                 distance_metric: str = 'cosine',
                 n_jobs: int = 1,
                 verbose: bool = False):

        self.word_embeddings = word_embeddings
        self.pooling = pooling
        self.fine_tune_mode = fine_tune_mode
        self.distance_metric = distance_metric
        self.n_jobs = n_jobs
        self.verbose = verbose


    def fit(self, X, y):

        tag_docs = self._create_tag_corpus(X, self._create_tag_docs(y))

        self.document_embedder_ = DocumentPoolEmbeddings(self.word_embeddings,
                                                         pooling=self.pooling,
                                                         fine_tune_mode=self.fine_tune_mode)

        if self.verbose:
            doc_iterator = tqdm(tag_docs, desc='Computing tag embeddings')
        else:
            doc_iterator = tag_docs

        self.tag_embeddings_ = []

        for doc in doc_iterator:
            doc_obj = Sentence(doc)
            self.document_embedder_.embed(doc_obj)
            self.tag_embeddings_.append(doc_obj.get_embedding().detach().numpy())

        self.tag_embeddings_ = np.array(self.tag_embeddings_)

        return self

    def predict(self, X: List[str], n_labels: int = 10) -> np.array:

        if not hasattr(self, 'tag_embeddings_'):
            raise NotFittedError


        if self.verbose:
            X_iterator = tqdm(X, desc='Computing embeddings for prediction samples')
        else:
            X_iterator = X

        X_embeddings = []

        for doc in X_iterator:
            doc_obj = Sentence(doc)
            self.document_embedder_.embed(doc_obj)
            X_embeddings.append(doc_obj.get_embedding().detach().numpy())


        nn = NearestNeighbors(metric=self.distance_metric, n_neighbors=n_labels, n_jobs=self.n_jobs)
        nn.fit(self.tag_embeddings_)

        y_pred = lil_matrix((len(X), self.tag_embeddings_.shape[0]), dtype='int8')

        for sample_ind, text_embedding in enumerate(X_embeddings):
            nearest_neighbors = nn.kneighbors([text_embedding])[1][0]
            y_pred[sample_ind, nearest_neighbors] = 1

        return y_pred.tocsr()

    def decision_function(self, X: List[str], n_labels: int = 10):

        if not hasattr(self, 'tag_embeddings_'):
            raise NotFittedError

        if self.verbose:
            X_iterator = tqdm(X, desc='Computing embeddings for prediction samples')
        else:
            X_iterator = X

        X_embeddings = []

        for doc in X_iterator:
            doc_obj = Sentence(doc)
            self.document_embedder_.embed(doc_obj)
            try:
                X_embeddings.append(doc_obj.get_embedding().detach().numpy())
            except RuntimeError as e:
                print('Could no compute embedding for sample inserting zero vector')
                # TODO give index of corrupted sample
                print(e)
                X_embeddings.append(np.zeros((self.tag_embeddings_[1], ), dtype=self.tag_embeddings_.dtype))

        nn = NearestNeighbors(metric=self.distance_metric, n_neighbors=n_labels, n_jobs=self.n_jobs)
        nn.fit(self.tag_embeddings_)

        y_pred = lil_matrix((len(X), self.tag_embeddings_.shape[0]), dtype='float')

        for sample_ind, sample_vec in enumerate(X_embeddings):
            distances, indices = nn.kneighbors([sample_vec])
            for distance, label_index in zip(distances, indices):
                y_pred[sample_ind, label_index] = distance

        return y_pred.tocsr()

    def log_decision_function(self, X: Iterable[str], n_labels: int = 10):
        if not hasattr(self, 'tag_embeddings_'):
            raise NotFittedError
        # TODO Uncomment this if sure that nothing will break
        distances = self.decision_function(X=X, n_labels=n_labels)
        log_distances = self._get_log_distances(distances)
        return log_distances

    def _get_log_distances(self, y_distances: csr_matrix, base=0.5) -> csr_matrix:
        """
        Returns the logarithmic version (base default: 0.5) of the distance matrix returned by TODO.
        This must be used in order to compute valid precision@k scores
        since small Distances should be ranked better than great ones.
        :param y_distances: sparse distance matrix (multilabel matrix with distances instead of binary indicators)
        :param base: base of the log function (must be smaller then one)
        :return: sparse matrix with the log values
        """

        log_y_distances = y_distances.tocoo()
        log_y_distances.data = np.log(log_y_distances.data) / np.log(base)
        return log_y_distances.tocsr()

    def _create_tag_corpus(self, X: np.array, tag_doc_idx: np.array) -> List[str]:
        """
        Creates the corpus used to train the tag embeddings.
        Each text associated with one tag is concatenated to one big document.
        :param X: Iterable of the texts as string
        :param tag_doc_idx: Mapping of each label to their associated texts
        :return: list of shape (n_tags,) containing the texts
        """
        tag_corpus = list()
        if self.verbose:
            print('Creating Tag-Doc Corpus')
            iterator = tqdm(tag_doc_idx)
        else:
            iterator = tag_doc_idx
        for indices in iterator:
            tag_corpus.append(" ".join(X[indices]))
        return tag_corpus

    def _create_tag_docs(self, y: csr_matrix) -> np.ndarray:
        """
        Creates a mapping of each tags and their associated texts.
        :param y: sparse label matrix
        :return: array of shape (n_labels,) containing the indices of each text connected to a label
        """
        self.classes_ = y.shape[1]

        if self.verbose:
            print('Sorting tag and docs')
            iterator = tqdm(y.T)
        else:
            iterator = y.T

        tag_doc_idx = list()
        for tag_vec in iterator:
            pos_samples = tag_vec.nonzero()[1]  # get indices of pos samples
            tag_doc_idx.append(pos_samples)
        return np.asarray(tag_doc_idx)

if __name__ == '__main__':

    X = np.array([
        'Das ist ein Auto',
        'Das ist ein Kino',
        'Das ist ein Buch',
        'Das ist ein Zug',
        'Das ist ein Flugzeug',
        'Das ist ein Computer'
    ])
    y = [
        [1, 2],
        [1, 4],
        [4, 5, 6],
        [1, 2, 3],
        [2, 5, 1],
        [9, 7]
    ]
    from sklearn.preprocessing import MultiLabelBinarizer
    from sklearn.model_selection import train_test_split
    from exmlc.metrics import sparse_average_precision_at_k

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mb = MultiLabelBinarizer(sparse_output=True)
    y_train = mb.fit_transform(y_train)
    y_test = mb.transform(y_test)

    import pandas as pd
    from exmlc.preprocessing import clean_string
    from exmlc.metrics import sparse_average_precision_at_k
    df = pd.read_csv('~/ba_arbeit/BA_Code/data/Stiwa/df_5.csv').dropna(subset=['keywords', 'text'])
    #df, df_remove = train_test_split(df, test_size=0.9, random_state=42)
    df.keywords = df.keywords.apply(lambda x: x.split('|'))
    df.text = df.text.apply(lambda x: clean_string(x, drop_stopwords=True))
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    X_train = df_train.text.to_numpy()
    X_test = df_test.text.to_numpy()

    mlb = MultiLabelBinarizer(sparse_output=True)
    y_train = mlb.fit_transform(df_train.keywords)
    y_test = mlb.transform(df_test.keywords)

    clf = FlairEmbeddingsClassifier(verbose=True, n_jobs=4)

    clf.fit(X_train, y_train)
    y_scores = clf.log_decision_function(X_test, n_labels=20)

    print(sparse_average_precision_at_k(y_test, y_scores, k=3))