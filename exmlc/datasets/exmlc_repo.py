from __future__ import annotations

from io import TextIOWrapper
from typing import *

from scipy.sparse import csr_matrix, lil_matrix


def load_fastxml_score_file(f: Union[str, TextIOWrapper]) -> csr_matrix:
    """
    Loads an score file returned by fastxml. Unused
    :param f:
    :return:
    """
    if isinstance(f, str):
        file = open(f, 'r')
    elif isinstance(f, TextIOWrapper):
        file = f
    else:
        raise TypeError(f'feature_file is type {type(f)} but should be either str or TextIOWrapper')

    header = file.readline()
    num_instances, num_labels = map(int, header.split(' '))
    y_scores = lil_matrix((num_instances, num_labels), dtype='float32')

    for row_index, row in enumerate(file):
        entries = row.split(' ')
        for entry in entries:
            label_index, value = entry.split(':')
            y_scores[row_index, int(label_index)] = float(value)

    file.close()

    return y_scores.tocsr()


def dump_slice_dataset(X: csr_matrix,
                       y: csr_matrix,
                       feat_file: Union[str, TextIOWrapper],
                       label_file: Union[str, TextIOWrapper]) -> None:
    """
    Dumps scipy matrices into format for slice. Unused
    """
    if isinstance(feat_file, str):
        feat_file = open(feat_file, 'w')
    elif isinstance(feat_file, TextIOWrapper):
        pass
    else:
        raise TypeError(f'feature_file is type {type(feat_file)} but should be either str or TextIOWrapper')

    if isinstance(label_file, str):
        label_file = open(label_file, 'w')
    elif isinstance(label_file, TextIOWrapper):
        pass
    else:
        raise TypeError(f'label_file is type {type(label_file)} but should be either str or TextIOWrapper')

    if X.shape[0] != y.shape[0]:
        raise Exception('X and y must have same shape')

    # 1. create sparse label file
    # format:
    # The first line of both the files contains the number of rows
    # the label file contains indices of active labels
    # and the corresponding value (always 1 in this case) starting from 0

    # write header
    label_header = f'{y.shape[0]} {y.shape[1]}\n'
    label_file.write(label_header)
    # write data
    for label_vector in y:
        label_idx = label_vector.nonzero()[1]
        line = f'{" ".join([f"{label_id}:1" for label_id in map(str, label_idx)])}\n'
        label_file.write(line)

    label_file.close()

    # 2. create dense feature file
    # format:
    # The first line of both the files contains the number of rows
    # For features, each line contains D (the dimensionality of the feature vectors), space separated, float values

    # write header
    feature_header = f'{X.shape[0]} {X.shape[1]}\n'
    feat_file.write(feature_header)
    # write data
    for feature_vector in X:
        line = f'{" ".join(map(str, [i if i > 0.0 else int(0) for i in feature_vector[0].toarray().ravel()]))}\n'
        feat_file.write(line)

    feat_file.close()

    return


def dump_xmlc_dataset(X: csr_matrix, y: csr_matrix, f: Union[str, TextIOWrapper]) -> None:
    """
    Dumps python data into xmlc format. Unused
    The data files for all the datasets are in the following sparse representation format:
        Header Line: Total_Points Num_Features Num_Labels
        1 line per datapoint : label1,label2,...labelk ft1:ft1_val ft2:ft2_val ft3:ft3_val .. ftd:ftd_val
    :param X:
    :param y:
    :param f:
    :return:
    """

    # check type of argument
    if isinstance(f, str):
        file = open(f, 'w')
    elif isinstance(f, TextIOWrapper):
        file = f
    else:
        raise TypeError(f'f is type {type(f)} but should be either str or TextIOWrapper')

    # create and write header
    header = f'{X.shape[0]} {X.shape[1]} {y.shape[1]}\n'
    file.write(header)

    # write data
    for index, (feature_vector, label_vector) in enumerate(zip(X, y)):
        labels = ','.join(map(str, label_vector.nonzero()[1]))

        features = [f'{str(ind)}:{str(feature_vector[0, ind])}' for ind in feature_vector.nonzero()[1]]
        features = " ".join(features)

        file.write(f'{" ".join((labels, features))}\n')

    file.close()

    return


def load_xmlc_dataset(f: Union[str, TextIOWrapper]) -> Tuple[csr_matrix, csr_matrix]:
    """
    Loads a extreme multilabel classification dataset provided by http://manikvarma.org/downloads/XC/XMLRepository.html
    Unused
    :param f: path to file as str or file like
    :return: X and y datasets in sparse representation
    """

    # check type of argument
    if isinstance(f, str):
        file = open(f, 'r')
    elif isinstance(f, TextIOWrapper):
        file = f
    else:
        raise TypeError(f'f is type {type(f)} but should be either str or TextIOWrapper')

    # read header
    header = file.readline()
    num_samples, num_features, num_labels = map(int, header.split(' '))

    # init matrices
    X = lil_matrix((num_samples, num_features), dtype='float32')
    y = lil_matrix((num_samples, num_labels), dtype='int8')

    # read data
    for row_index, line in enumerate(file):
        # strip leading whitespaces and split line
        entries = line.lstrip().split(' ')
        # read labels
        # 1. check if there are any labels at this point
        if ':' not in entries[0]:
            label_idx = entries.pop(0).split(',')
            label_idx = [int(i) for i in label_idx]
            y[row_index, label_idx] = 1
        # read features
        for entry in entries:
            feature_index, feature_value = entry.split(':')
            feature_index, feature_value = int(feature_index), float(feature_value)
            X[row_index, feature_index] = feature_value

    file.close()

    return X.tocsr(), y.tocsr()
