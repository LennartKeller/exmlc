
import re
import string
from collections import OrderedDict

from stop_words import get_stop_words


def clean_string(x: string, drop_stopwords: bool = True) -> str:
    """
    Cleans the the given strings.
    Removes punctuation marks.
    Removes some specials structures which occur in the dataset.
    :param x: string to clearn
    :param drop_stopwords: True if stopwords should be removed, else False
    :return: cleaned string
    """
    stop_words = get_stop_words('de')

    removal_regex = OrderedDict([
        # remove aktenzeichen
        (re.compile(r'\(az.+\)', re.MULTILINE | re.IGNORECASE), ' '),
        # remove cms product refereneces
        (re.compile(r'\[\[. +?]]', re.MULTILINE | re.IGNORECASE), ' '),
        # remove prices
        (re.compile(r'\d+ ?\d+ Euro', re.MULTILINE | re.IGNORECASE), ' '),
        # remove full urls
        (re.compile(r'www\..+\..{2,3}', re.MULTILINE | re.IGNORECASE), ' '),
        # remove in text links
        (re.compile(r'\w+\.(?:de|com|org)', re.MULTILINE | re.IGNORECASE), ' '),
        # remove degree celsius
        (re.compile(r'°c', re.MULTILINE | re.IGNORECASE), ' '),
        # remove digits
        (re.compile(r'\d+', re.MULTILINE | re.IGNORECASE), ' '),

    ])
    marks_to_remove = string.punctuation + "„“–’"

    if x:
        for key in removal_regex:
            x = re.sub(key, removal_regex[key], x)
        # remove punctuation marks
        # 1. replace - marks with empty string to glue words back together
        x = x.replace('\xad', '')
        x = x.replace('-', ' ')
        # 2. replace every other punctuation mark with whitespace to prevent glueing together
        x = x.translate(str.maketrans(marks_to_remove, ' ' * len(marks_to_remove)))
        # normalize whitespaces and if stated drop stopwords along the way
        if drop_stopwords:
            x = [i for i in x.split() if i not in stop_words]
        else:
            x = x.split()
        xt = " ".join(x)
    else:
        xt = ''

    return xt


def tokenize_count(s: str) -> int:
    """
    Tokenizes the given strings to count the number of words.
    :param s:
    :return: number of words
    """
    s = s.translate(str.maketrans('', '', string.punctuation + "„“–"))
    return len(re.split(r'\W+', s))