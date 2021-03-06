from __future__ import annotations

import numpy as np

from sklearn.base import BaseEstimator
from gensim.models import Word2Vec
from scipy.sparse import csr_matrix, lil_matrix
from typing import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from gensim.models.fasttext import FastText
from gensim.models.word2vec import Word2Vec
from sklearn.exceptions import NotFittedError
import logging
from tqdm import tqdm
from gensim.similarities.docsim import WmdSimilarity

class Word2VecTagEmbeddingClassifier(BaseEstimator):

    def __init__(self,
                 embedding_dim: int = 300,
                 min_count: int = 1,
                 window_size: int = 5,
                 epochs: int = 10,
                 model: str = 'doc2vec',
                 distance_metric: str = 'cosine',
                 tfidf_weighting: bool = True,
                 pooling_func: callable = lambda x: np.mean(x, axis=0),  # column wise average
                 n_jobs: int = 1,
                 verbose: bool = False
                 ) -> None:

        self.embedding_dim = embedding_dim
        self.min_count = min_count
        self.window_size = window_size
        self.epochs = epochs
        self.model = model
        self.distance_metric = distance_metric
        self.tfidf_weighting = tfidf_weighting
        self.pooling_func = pooling_func
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X: np.array, y: csr_matrix):



        if self.verbose:
            #logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
            # TODO revert this
            pass

        X_splitted = np.array([s.split() for s in X])
        #docs = [TaggedDocument(words=tokens, tags=[index]) for index, tokens in enumerate(X_splitted)]

        if self.model.lower() == 'fasttext':
            self.wv_model_ = FastText(
                sentences=X_splitted.tolist(),
                size=self.embedding_dim,
                iter=self.epochs,
                min_count=self.min_count,
                window=self.window_size,
                workers=self.n_jobs
            )
        elif self.model.lower() == 'doc2vec':

            self.wv_model_ = Word2Vec(
                sentences=X_splitted.tolist(),
                size=self.embedding_dim,
                iter=self.epochs,
                min_count=self.min_count,
                window=self.window_size,
                workers=self.n_jobs,
            )

        else:
            raise NotImplementedError

        tag_doc_mapping = self._create_tag_docs(y)

        if self.tfidf_weighting:
            self.tfidf_ = TfidfVectorizer()
            self.texts_tfidf_ = self.tfidf_.fit_transform(X)

        self.tag_embeddings_ = np.empty((y.shape[1], self.embedding_dim), dtype='float64')

        if self.verbose:
            tac_doc_iterator = tqdm(enumerate(tag_doc_mapping), desc='Computing tag embeddings')
        else:
            tac_doc_iterator = enumerate(tag_doc_mapping)
        for tag_id, texts_idx in tac_doc_iterator:
            # will be of shape(n_texts, embedding_dim)
            tag_word_embeddings = []
            for text_ind in texts_idx:
                for token in list(set(X_splitted[text_ind])):
                    try:
                        word_embedding = self.wv_model_.wv[token]
                    except KeyError:
                        # if words occur that are ignored due to min_count
                        continue
                    if self.tfidf_weighting:
                        token_ind = self.tfidf_.vocabulary_.get(token, -1)
                        if token_ind > -1:
                            tfidf_value = self.texts_tfidf_[text_ind, token_ind]
                            word_embedding = word_embedding * tfidf_value
                    tag_word_embeddings.append(word_embedding)

            self.tag_embeddings_[tag_id] = self.pooling_func(tag_word_embeddings)
        return self

    def predict(self, X: List[str], n_labels: int = 10) -> np.array:

        if not hasattr(self, 'tag_embeddings_'):
            raise NotFittedError

        X_splitted = [s.split() for s in X]
        X_embeddings = []
        for text in X_splitted:
            text_word_embeddings = []
            for token in text:
                try:
                    word_embedding = self.wv_model_.wv[token]
                except KeyError:
                    continue
                text_word_embeddings.append(word_embedding)
            X_embeddings.append(self.pooling_func(text_word_embeddings))

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

        X_splitted = [s.split() for s in X]
        X_embeddings = []
        for text in X_splitted:
            text_word_embeddings = []
            for token in text:
                try:
                    word_embedding = self.wv_model_.wv[token]
                except KeyError:
                    continue
                text_word_embeddings.append(word_embedding)
            X_embeddings.append(self.pooling_func(text_word_embeddings))

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
            t = tag_vec.nonzero()
            pos_samples = tag_vec.nonzero()[1]  # get indices of pos samples
            tag_doc_idx.append(pos_samples)
        return np.asarray(tag_doc_idx)


    def wmd(self, X_train, y_train, X_test, n_labels: int = 10, n_ev: int = 2):
        """
        Compute docs similarity scores using the word mover distance (Kusner et. al, 2015)
        Since this is computationally expensive because every docs from test set has to be compared to each doc
        in train set the centroid optimatzion as described by Kusner et. al is used.
        :param X_train:
        :param X_test:
        :param n_labels: number of desired label to predict
        :param n_ev: factor for size of search space
        the search space for wdm which is precomputed will be of size n_labels * n_ev
        :return:
        """

        # Compute and store mean doc embedding for each doc in train-set
        # => is done while fitting so we can use the self.tag_embeddings_ attribute
        # For each sample in X_test
        X_embeddings = []  # store mean doc embeddings for X_test
        for x_sample in X_test:
            x_sample = x_sample.split()
            # Compute mean doc embedding for test doc
            x_embeddings = []
            for token in x_sample:
                try:
                    word_embedding = self.wv_model_.wv[token]
                except KeyError:
                    continue
                x_embeddings.append(word_embedding)
            X_embeddings.append(np.mean(x_embeddings, axis=0))

        nn = NearestNeighbors(n_neighbors=n_labels * n_ev).fit(self.tag_embeddings_)

        X_nearest_tags = nn.kneighbors(X_embeddings)[1]  # indices of most simiilar tag docs

        # recompute tag docs
        tag_doc_idx = self._create_tag_docs(y_train)
        tag_docs = [[] for _ in range(len(tag_doc_idx))]
        for doc_idx, entry in zip(tag_doc_idx, tag_docs):
            for doc_id in doc_idx:
                entry.extend(X_train[doc_id].split())

        tag_docs = np.array(tag_docs)

        results = []
        y_pred = lil_matrix((X_test.shape[0], y_train.shape[1]), dtype='int8')
        if self.verbose:
            iterator = tqdm(enumerate(zip(X_nearest_tags, X_test)), desc='Computing wdm distances')
        else:
            iterator = enumerate(zip(X_nearest_tags, X_test))
        for sample_ind, (nearest_tag_doc_idx, x_sample) in iterator:  # TODO fix typo in loop var
            wmd = WmdSimilarity(tag_docs[nearest_tag_doc_idx], self.wv_model_, num_best=n_labels)
            sim_mat = wmd[x_sample.split()]
            results.append(nearest_tag_doc_idx[[i[0] for i in sim_mat]])
            y_pred[sample_ind, nearest_tag_doc_idx[[i[0] for i in sim_mat]]] = 1

        return y_pred


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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mb = MultiLabelBinarizer(sparse_output=True)
    y_train = mb.fit_transform(y_train)
    y_test = mb.transform(y_test)

    import pandas as pd
    from exmlc.preprocessing import clean_string
    from exmlc.metrics import sparse_average_precision_at_k
    df = pd.read_csv('~/ba_arbeit/BA_Code/data/Stiwa/df_5.csv').dropna(subset=['keywords', 'text'])
    df, df_remove = train_test_split(df, test_size=0.99, random_state=42)
    df.keywords = df.keywords.apply(lambda x: x.split('|'))
    df.text = df.text.apply(lambda x: clean_string(x, drop_stopwords=True))
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    X_train = df_train.text.to_numpy()
    X_test = df_test.text.to_numpy()

    mlb = MultiLabelBinarizer(sparse_output=True)
    y_train = mlb.fit_transform(df_train.keywords)
    y_test = mlb.transform(df_test.keywords)

    clf = Word2VecTagEmbeddingClassifier(embedding_dim=300,
                                         min_count=5,
                                         model='doc2vec',
                                         epochs=20,
                                         window_size=5,
                                         tfidf_weighting=False,
                                         verbose=True,
                                         n_jobs=4)

    clf.fit(X_train, y_train)
    #y_scores = clf.log_decision_function(X_test, n_labels=10)
    y_pred = clf.wmd(X_train, y_train, X_test)
    #print(sparse_average_precision_at_k(y_test, y_scores, k=3))
    from sklearn.metrics import f1_score
    print(f1_score(y_test, y_pred, average='macro'))
