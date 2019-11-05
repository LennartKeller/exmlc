from __future__ import annotations
from typing import *
from sklearn.base import BaseEstimator
from scipy.sparse import csr_matrix
from sklearn.utils import check_X_y
import numpy as np
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from sklearn.exceptions import NotFittedError
from sklearn.neighbors import NearestNeighbors
import logging
from tqdm import tqdm

class TagEmbeddingClassifier(BaseEstimator):

    def __init__(self,
                 embedding_dim: int = 300,
                 window_size: int = 5,
                 min_count: int = 2,
                 pooling:str = 'max',
                 epochs: int = 10,
                 distance_metric: str = 'cosine',
                 n_jobs: int = 1,
                 verbose: bool = False):

        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.min_count = min_count
        self.pooling = pooling
        self.epochs = epochs
        self.distance_metric = distance_metric
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X: Union[np.ndarray, csr_matrix], y: csr_matrix) -> TagEmbeddingClassifier:

        # create tag_docs (each text assiganed to the tag)
        self.tag_doc_idx_ = self._create_tag_docs(X, y)
        # create corpus
        self.tag_corpus = self._create_tag_corpus(X, self.tag_doc_idx_)

        # created TaggedDocuments for Doc2Vec gensim
        tagged_docs = [TaggedDocument(words=doc.split(),tags=[tag_id]) for tag_id, doc in enumerate(self.tag_corpus)]
        if self.verbose:
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

        # train model
        self.doc2vec_model_ = Doc2Vec(tagged_docs,
                        vector_size=self.embedding_dim,
                        window=self.window_size,
                        min_count=self.min_count,
                        workers=self.n_jobs,
                        epochs=self.epochs)

        self.doc_embeddings_ = self.doc2vec_model_.docvecs.vectors_docs.copy()

        return self

    def predict(self, X: Iterable[str], n_labels=10) -> np.array:
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
        return np.asarray(tags, dtype='int64')





    def _create_tag_docs(self, X: Union[np.ndarray, csr_matrix], y: csr_matrix) -> np.ndarray:
        self.classes_ = y.shape[1]
        tag_doc_idx = list()
        if self.verbose:
            print('Sorting tag and docs')
            iterator = tqdm(y.T)
        else:
            iterator = y.T
        for tag_vec in iterator:
            pos_samples = tag_vec.nonzero()[1]
            tag_doc_idx.append(pos_samples)
        return np.asarray(tag_doc_idx)

    def _create_tag_corpus(self, X: np.array, tag_doc_idx: np.array) -> List[str]:
        tag_corpus = list()
        if self.verbose:
            print('Creating Tag-Doc Corpus')
            iterator = tqdm(tag_doc_idx)
        else:
            iterator = tag_doc_idx
        for indices in iterator:
            tag_corpus.append(" ".join(X[indices]))
        return np.asarray(tag_corpus)

    def _tagged_document_generator(self, tag_corpus: np.array) -> Generator[TaggedDocument, None, None]:
        if self.verbose:
            print('Preparing Tagged Documents for Doc2Vec Model')
            iterator = tqdm(enumerate(tag_corpus))
        else:
            iterator = enumerate(tag_corpus)
        for tag_id, doc in iterator:
            yield TaggedDocument(words=doc.split(' '), tags=[tag_id])

    def _infer_new_docs(self, X: Iterable[str]) -> np.ndarray:
        # infer new doc to obtain vector
        sample_doc_embeddings = []
        for sample in X:
            sample_doc_embeddings.append(self.doc2vec_model_.infer_vector(sample.split()))
        return sample_doc_embeddings


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
        [1,2],
        [1,4],
        [4,5,6],
        [1,2,3],
        [2,5,1],
        [9,7]
    ]
    from sklearn.preprocessing import MultiLabelBinarizer
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mb = MultiLabelBinarizer(sparse_output=True)
    y_train = mb.fit_transform(y_train)
    y_test = mb.transform(y_test)

    tec = TagEmbeddingClassifier()

    print('Start training')
    tec.fit(X_train, y_train)
    print(tec)
    print('Start predicting')
    pred = tec.predict(X_test, n_labels=2)