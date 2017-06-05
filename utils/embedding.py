__author__ = 'ando'
import itertools
from functools import partial
import logging as log
import torch as t
import numpy as np
from scipy.special import expit as sigmoid

log.basicConfig(format='%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s', level=log.DEBUG)

long_tensor = t.LongTensor

def batch_generator(iterable, chunksize):
    """
    Return elements from the iterable in `chunksize`-ed lists. The last returned
    element may be smaller (if length of collection is not divisible by `chunksize`).

    >>> print(list(grouper(range(10), 3)))
    [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]

    """

    it = iter(iterable)

    b_input, b_output = map(list, zip(*itertools.islice(it, int(chunksize))))
    try:
        while (b_input, b_output):
            yield long_tensor(b_input), long_tensor(b_output)
            b_input, b_output = map(list, zip(*itertools.islice(it, int(chunksize))))
    except ValueError as e:
        log.info("end of dataset")


    # batch_input = [[]]
    # batch_output = [[]]
    # while input, outputs in itertools.islice(it, int(chunksize)):
    #     batch_input[0].append(input)
    #     batch_output[0].append(outputs)
    # yield long_tensor(batch_input.pop()), long_tensor(batch_output.pop())

def prepare_sentences(model, examples, transfer_fn):
    '''
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
    def __init__(self, corpus, n):
        self.corpus = corpus
        self.n = n
        self.total_word = self.corpus.shape[0] * self.corpus.shape[1] * self.n
        self.total_examples = len(self.corpus) * self.n

    def __iter__(self):
        for _ in range(self.n):
            for document in self.corpus:
                yield document


# def generate_batch(X, batch_size, num_skips=2, window_size=5):
#     """
#     batch generation for o1 and o2
#     :param X: input data
#     :param batch_size: size of the batch
#     :param num_skips: How many times the same input is used to generate a label
#     :return:
#     """
#     global data_index
#     assert batch_size % num_skips == 0
#     assert num_skips <= 2 * window_size
#     batch = torch.zeros(batch_size).type(short_dtype)
#     labels = torch.zeros(batch_size, 1).type(short_dtype)
#
#     span = 2 * window_size + 1  # [ skip_window target skip_window ]
#     buffer = deque(maxlen=span)
#     for _ in range(span):
#         buffer.append(X[data_index])
#         data_index = (data_index + 1) % len(X)
#     for i in range(batch_size // num_skips):
#         target = window_size  # target label at the center of the buffer
#         targets_to_avoid = [self.window_size]
#         for j in range(num_skips):
#             while target in targets_to_avoid:
#                 target = random.randint(0, span - 1)
#             targets_to_avoid.append(target)
#             batch[i * num_skips + j] = buffer[self.window_size]
#             labels[i * num_skips + j, 0] = buffer[target]
#         buffer.append(X[data_index])
#         data_index = (data_index + 1) % len(X)
#     # Backtrack a little bit to avoid skipping words in the end of a batch
#     data_index = (data_index + len(X) - span) % len(X)
#     return batch, labels

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
