__author__ = 'ando'
import itertools
from functools import partial
import logging as log
import torch as t
import numpy as np
from scipy.special import expit as sigmoid

log.basicConfig(format='%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s', level=log.DEBUG)

def batch_generator(iterable, batch_size, long_tensor=t.LongTensor):
    """
    Return elements from the iterable in `chunksize`-ed lists. The last returned
    element may be smaller (if length of collection is not divisible by `chunksize`).

    >>> print(list(grouper(range(10), 3)))
    [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]

    :param iterable: iterator on which create the data
    :param batch_size: size of the batch
    :param long_tensor: type of the tensor created
    :return:
    """

    it = iter(iterable)

    b_input, b_output = map(list, zip(*itertools.islice(it, int(batch_size))))
    try:
        while (b_input, b_output):
            yield long_tensor(b_input), long_tensor(b_output)
            b_input, b_output = map(list, zip(*itertools.islice(it, int(batch_size))))
    except ValueError as e:
        log.info("end of dataset")


def prepare_sentences(model, examples, transfer_fn):
    '''
    Convert the paths from node_id to node_index and perform subsampling if setted in the model

    :param model: current deprecated_model containing the vocabulary and the index
    :param examples: list of the example. we have to translate the node to the appropriate index and apply the dropout
    :param transfer_fn: function used to translate the output_labels
    :return: generator of the paths according to the dropout probability and the correct index
    '''
    for input_labels, out_labels in examples:
        if model.vocab[input_labels].sample_probability >= 1.0 or model.vocab[input_labels].sample_probability >= np.random.random_sample():
            yield model.vocab[input_labels].index, transfer_fn(out_labels)
            # [deprecated_model.vocab[node].index for node in out_labels]
        else:
            continue

class RepeatCorpusNTimes():
    """
    Helper class used to repeat the dataset n times
    """
    def __init__(self, corpus, n):
        self.corpus = corpus
        self.n = n
        self.total_word = self.corpus.shape[0] * self.corpus.shape[1] * self.n
        self.total_examples = len(self.corpus) * self.n

    def __iter__(self):
        for _ in range(self.n):
            for document in self.corpus:
                yield document

class Vocab(object):
    """A single vocabulary item, used internally for constructing binary trees (incl. both word leaves and inner nodes)."""
    def __init__(self, **kwargs):
        self.count = 0
        self.__dict__.update(kwargs)

    def __lt__(self, other):  # used for sorting in a priority queue
        return self.count < other.count

    def __str__(self):
        vals = ['{}:{}'.format(key, self.__dict__[key]) for key in sorted(self.__dict__) if not key.startswith('_')]
        return "{}\t<{}>".format(self.__class__.__name__, ', '.join(vals))
