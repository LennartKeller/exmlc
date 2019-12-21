from __future__ import annotations

import logging
from typing import *

import numpy as np
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


class TagEmbeddingClassifier(BaseEstimator):
    """
    This Classifiers computes "tag-embeddings" by using the raw texts assigned to the tags
    as described by Chen et. al (2017)
    Link: https://pdfs.semanticscholar.org/2301/98d3b8ad550aaa77d28155091cc8a3d4032d.pdf

    Please note that this classifier does not implement the main method described in this paper
    but the much simpler baseline model which they use for performance evaluation of their postulated method.

    This model uses the gensim library and its implementation
    of doc2vec (Mikolov et. al: https://arxiv.org/pdf/1405.4053v2.pdf) for computing the tag-embeddings
    and the NearestNeighbors model from scikit-learn for getting the most similar tags at prediction.

    In theory this model is highly flexible because it not only has a lot of parameters which can be fine tuned
    But in practice as of now the performance is rather poor (yielding often not more than 25% precision@3).

    For more information about the parameters please refer to the constructor method.

    Training:
    Each text which is assigned to a specific tag from the train data
    is concatenated and then document embeddings are trained on these collections of text.
    So these each of this document embedding vectors represents a tag and therefore will be called tag-embedding.

    Prediction:
    Document embeddings for the new texts are computed using the trained embedding-model.
    Then a k-nearest neighbor search is performed in order to find the most similar tags-embeddings.

    In general embeddings are a technique to embed objects into a vector space of a distinct dimension
    such that similar objects appear nearby in the vector space.
    word-embeddings are a unsupervised learning techniques which aims at computing vector representations for words.
    The goal is that the vectors hold some sort of semantic information so that semantically related words also appear
    nearby in the vector space. Hence a single word vector alone does not provide any useful or
    even meaningful information about the semantic dimensions of a word
    but a set of words is needed to evaluate the embeddings.

    The idea to model language in a vector space had its first appearance in the field of information retrieval where
    documents where modelled as vectors containing word frequencies in order to find documents which share similar
    contents. Later this approach was advanced by latent semantic indexing which does not try to find similarities
    between documents based on their words but based on abstract underlying topics which can be extracted using
    mathematical methods such as single value decomposition.

    Th
    """

    def __init__(self,
                 embedding_dim: int = 300,
                 window_size: int = 5,
                 min_count: int = 2,
                 epochs: int = 10,
                 additional_doc2vec_params: dict = None,
                 distance_metric: str = 'cosine',
                 n_jobs: int = 1,
                 verbose: bool = False):

        """
        Constructor of the TagEmbedding model class.
        :param embedding_dim: the number of dimension a computed tag embedding will have
        :param window_size: the size of the sliding window for computing the word embeddings
        that will be used to compute the tag embeddings
        :param min_count: minimum number of occurrences a word must have in order
        to be used for computing the tag embeddings
        :param epochs: number of epochs (complete iterations over the whole train data)
        when computing the tag embeddings
        :param distance_metric: the distane metric for finding similar tags in the vector space
        :param n_jobs: number of cores to use for training the model (-1 means all available cores)
        :param verbose: whether or not information while training should be printed
        """

        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.min_count = min_count
        self.epochs = epochs
        self.distance_metric = distance_metric
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.additional_doc2vec_params = additional_doc2vec_params

    def fit(self, X: np.array, y: csr_matrix) -> TagEmbeddingClassifier:
        """
        Fits the model.
        First the train data will grouped by the tags.
        Then the embeddings are trained.
        The trained embeddings are bound to the doc_embeddings_ attribute
        The number of tags in the training set will be bound to n_tags_
        :param X: numpy array of documents as strings
        :param y: label matrix in sparse format
        :return: fitted instance of itself
        """

        self.n_tags_ = y.shape[1]

        # create tag_docs (each text assiganed to the tag)
        # self.tag_doc_idx_ = self._create_tag_docs(X, y)
        # create corpus
        # self.tag_corpus = self._create_tag_corpus(X, self.tag_doc_idx_)
        # created TaggedDocuments for Doc2Vec gensim
        tagged_docs = list(self._tagged_document_generator(self._create_tag_corpus(X, self._create_tag_docs(y))))
        if self.verbose:
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

        # train model
        if not self.additional_doc2vec_params:
            self.doc2vec_model_ = Doc2Vec(tagged_docs,
                                          vector_size=self.embedding_dim,
                                          window=self.window_size,
                                          min_count=self.min_count,
                                          workers=self.n_jobs,
                                          epochs=self.epochs)
        else:
            self.doc2vec_model_ = Doc2Vec(tagged_docs,
                                          vector_size=self.embedding_dim,
                                          window=self.window_size,
                                          min_count=self.min_count,
                                          workers=self.n_jobs,
                                          epochs=self.epochs,
                                          **self.additional_doc2vec_params)

        self.doc_embeddings_ = self.doc2vec_model_.docvecs.vectors_docs.copy()

        return self

    def predict(self, X: Iterable[str], n_labels=10) -> np.array:
        """
        Predicts n labels for every document in X.
        Therefore a embedding for each doc in X is computed using gensims Doc2Vec class.
        After that a k nearest neighbor search is used to find the n most close tag embedding in the embedding space.
        :param X: new Document to be tagged
        :param n_labels: the desired number of tags for each text
        :return: a binary label matrix of shape (len(X), self.n_tags)
        """
        if not hasattr(self, 'doc_embeddings_'):
            raise NotFittedError
        new_doc_embeddings = self._infer_new_docs(X)
        knn = NearestNeighbors(n_neighbors=n_labels, metric=self.distance_metric)
        knn.fit(self.doc_embeddings_)
        X_nearest_neighbors = []
        for emb in new_doc_embeddings:
            nearest_neighbors = knn.kneighbors([emb])[1][0]
            tags = [self.doc2vec_model_.docvecs.index_to_doctag(i).item() for i in nearest_neighbors]
            X_nearest_neighbors.append(tags)

        # create multilabel binary sparse matrix
        result = lil_matrix((X.shape[0], self.n_tags_), dtype='int8')
        for sample_ind, tag_idx in enumerate(X_nearest_neighbors):
            result[sample_ind, tag_idx] = 1
        return result.tocsr()

    def decision_function(self, X: Iterable[str], n_labels: int = 10):
        """
        Returns the distances to each predicted tag.
        This does mostly the same as the predict method but instead of returning a binary label matrix
        it returns a label matrix where each entry for a predicted tag is the distance
        between the embedding document and the tag embedding.
        Please note that only the distances between the n_labels most similar tags are computed.
        :param X: new Document to be tagged
        :param n_labels: the desired number of tags for each text
        :return: a label matrix of shape (len(X), self.n_tags) with the distances instead of binary indicators
        """
        if not hasattr(self, 'doc_embeddings_'):
            raise NotFittedError

        new_doc_embeddings = self._infer_new_docs(X)
        knn = NearestNeighbors(n_neighbors=n_labels, metric=self.distance_metric, n_jobs=self.n_jobs)
        knn.fit(self.doc_embeddings_)
        X_nearest_neighbors = []
        for emb in new_doc_embeddings:
            nearest_neighbors = knn.kneighbors([emb])
            distances, idx = nearest_neighbors
            distances, idx = distances[0], idx[0]
            tags = [self.doc2vec_model_.docvecs.index_to_doctag(i).item() for i in idx]
            idx_tags = list(zip(distances, tags))
            X_nearest_neighbors.append(idx_tags)
        result = lil_matrix((X.shape[0], self.n_tags_), dtype='float64')
        for sample_ind, entry in enumerate(X_nearest_neighbors):
            for tag_distance, tag_ind in entry:
                result[sample_ind, tag_ind] = tag_distance
        return result.tocsr()

    def log_decision_function(self, X: Iterable[str], n_labels: int = 10):
        if not hasattr(self, 'doc_embeddings_'):
            raise NotFittedError
        # TODO Uncomment this if sure that nothing will break
        distances = self.decision_function(X=X, n_labels=n_labels)
        log_distances = self._get_log_distances(distances)
        return log_distances

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

    def _tagged_document_generator(self, tag_corpus: list) -> Generator[TaggedDocument, None, None]:
        """
        Generator yielding the tagged document object required by gensim.
        :param tag_corpus: the corpus of tags and their texts
        :return: TaggedDocument objects
        """
        if self.verbose:
            print('Preparing Tagged Documents for Doc2Vec Model')
            iterator = tqdm(enumerate(tag_corpus))
        else:
            iterator = enumerate(tag_corpus)
        for tag_id, doc in iterator:
            yield TaggedDocument(words=doc.split(), tags=[tag_id])

    def _infer_new_docs(self, X: Iterable[str]) -> np.ndarray:
        """
        Computes a embedding for new document at prediction.
        :param X: the documents to predict tags
        :return: Iterable of new embeddings for each document in X
        """
        sample_doc_embeddings = []
        for sample in X:
            sample_doc_embeddings.append(self.doc2vec_model_.infer_vector(sample.split(), epochs=self.epochs))
        return sample_doc_embeddings  # TODO can this be casted to a np.array?

    def _get_log_distances(self, y_distances: csr_matrix, base=0.5) -> csr_matrix:
        """
        Returns the logarithmic version (base default: 0.5) of the distance matrix returned by TagEmebddingClassifier.
        This must be used in order to compute valid precision@k scores
        since small Distances should be ranked better than great ones.
        :param y_distances: sparse distance matrix (multilabel matrix with distances instead of binary indicators)
        :param base: base of the log function (must be smaller then one)
        :return: sparse matrix with the log values
        """

        log_y_distances = y_distances.tocoo()
        log_y_distances.data = np.log(log_y_distances.data) / np.log(base)
        return log_y_distances.tocsr()


if __name__ == '__main__':  # used for debugging and testing ...

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

    tec = TagEmbeddingClassifier(min_count=5, window_size=5, embedding_dim=300, epochs=20, n_jobs=4)

    print('Start training')
    tec.fit(X_train, y_train)
    print(tec)
    print('Start predicting')
    pred = tec.predict(X_test, n_labels=20)
    print(f1_score(y_test, pred, average='macro'))

