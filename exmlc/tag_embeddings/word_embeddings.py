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
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

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
                workers=self.n_jobs
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

    def predict(self, X: List[str], n_labels: int =10) -> np.array:

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
                    # TODO may some statistics here?
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
    from sklearn.metrics import f1_score
    df = pd.read_csv('~/ba_arbeit/BA_Code/data/Stiwa/df_5.csv').dropna(subset=['keywords', 'text'])
    df, df_remove = train_test_split(df, test_size=0.9, random_state=42)
    df.keywords = df.keywords.apply(lambda x: x.split('|'))
    df.text = df.text.apply(lambda x: clean_string(x, drop_stopwords=True))
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    X_train = df_train.text.tolist()
    X_test = df_test.text.tolist()

    mlb = MultiLabelBinarizer(sparse_output=True)
    y_train = mlb.fit_transform(df_train.keywords)
    y_test = mlb.transform(df_test.keywords)

    clf = Word2VecTagEmbeddingClassifier(embedding_dim=300, min_count=5,
                                         epochs=20,
                                         window_size=5, tfidf_weighting=True, verbose=True, n_jobs=4)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test, n_labels=5)

    print(f1_score(y_test, y_pred, average='macro'))



