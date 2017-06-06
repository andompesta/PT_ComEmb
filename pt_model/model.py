__author__ = 'ando'
import logging as log
import numpy as np
import pickle
from os.path import exists
from os import makedirs
from utils.embedding import Vocab
from utils.IO_utils import load_ground_true
import torch as t
import torch.nn as nn
from torch.nn import Parameter

log.basicConfig(format='%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s', level=log.DEBUG)

float_tensor = None

class ComEModel(object):
    '''
    class that keep track of all the parameters used during the learning of the embedding.
    '''


    def __init__(self, nodes_degree,
                 size=2,
                 down_sampling=0,
                 seed=1,
                 path_labels='data/',
                 input_file=None,
                 f_type=t.FloatTensor):
        '''
        :param nodes_degree: Dict with node_id: degree of node
        :param size: projection space
        :param downsampling: perform downsampling of common node
        :param seed: seed for random function
        :param index2node: index between a node and its representation

        :param path_labels: location of the file containing the ground true (label for each node)
        :param input_file: name of the file containing the ground true (label for each node)
        :return:
        '''
        global float_tensor
        float_tensor = f_type

        self.down_sampling = down_sampling
        self.seed = seed

        if size % 4 != 0:
            log.warn("consider setting layer size to a multiple of 4 for greater performance")
        self.layer1_size = int(size)

        if nodes_degree is not None:
            self.build_vocab_(nodes_degree)
            self.ground_true, self.k = load_ground_true(path=path_labels, file_name=input_file)
            # inizialize node and context embeddings
            self.reset_weights()
            self.compute_negative_sampling_weight()
        else:
            log.info("Model not initialized")

        

    def build_vocab_(self, vocab):
        """
        Build vocabulary from a sequence of paths (can be a once-only generator stream).
        Sorted by node id
        """
        # assign a unique index to each node
        self.vocab = {}

        for node_idx, (node, count) in enumerate(sorted(vocab.items(), key=lambda itm: itm[0])):
            v = Vocab()
            v.count = count
            v.index = node_idx
            # self.index2node.append(node)
            self.vocab[node] = v
        assert min(self.vocab.keys()) == 1
        self.precalc_sampling()


    def precalc_sampling(self):
        '''
            Peach vocabulary item's threshold for sampling
        '''

        if self.down_sampling:
            print("frequent-node down sampling, threshold %g; progress tallies will be approximate" % (self.down_sampling))
            total_nodes = sum(v.count for v in self.vocab.values())
            threshold_count = float(self.down_sampling) * total_nodes

        for v in self.vocab.values():
            prob = (np.sqrt(v.count / threshold_count) + 1) * (threshold_count / v.count) if self.down_sampling else 1.0
            v.sample_probability = min(prob, 1.0)


    def reset_weights(self):
        """Reset all projection weights to an initial (untrained) state, but keep the existing vocabulary."""
        self.size = len(self.vocab)
        self.node_embedding = nn.Embedding(self.size, self.layer1_size)
        self.node_embedding.weight = Parameter(float_tensor(self.size, self.layer1_size).uniform_(-1, 1))


        self.context_embedding = nn.Embedding(self.size, self.layer1_size)
        self.context_embedding.weight = Parameter(t.zeros(self.size, self.layer1_size).type(float_tensor))

    def compute_negative_sampling_weight(self, power=0.75):
        """
        Compute the negative sampling probability
        :param power: normalization probability
        :return: 
        """
        log.info("constructing a table with noise distribution from %i nodes" % self.size)

        self.sampling_weight = np.zeros(self.size)
        if not self.size:
            log.error("empty vocabulary in, is this intended?")
            return

        train_nodes_pow = float(sum([self.vocab[node].count ** power for node in self.vocab]))          # sum for normalization
        for node_id, node in self.vocab.items():
            prob = node.count ** power / train_nodes_pow                                    # compute negative sampling prob for each word
            assert prob > 0., "each sampling prob should be greater than 0"
            self.sampling_weight[node.index] = prob

        # NORMALIZING
        # TODO: check if it is needed the second normalization
        # sum_weights = sum(self.sampling_weight)
        # self.sampling_weight = [w / sum_weights for w in self.sampling_weight]

    def negative_sample(self, n_samples):
        """
        draws a sample from classes based on weights
        :param n_samples: number of negative sample
        :return: 
        """
        draw = np.random.choice(self.size, n_samples, p=self.sampling_weight)
        return np.array(draw)

    def get_node_embedding(self):
        return self.node_embedding.weight.data.cpu().numpy()




    def save(self, path='data', file_name=None):
        if not exists(path):
            makedirs(path)

        with open(path + '/' + file_name + '.bin', 'wb') as file:
            pickle.dump(self.__dict__, file)

    @staticmethod
    def load_model(path='data', file_name=None):
        with open(path + '/' + file_name + '.bin', 'rb') as file:
            model = {}
            model.__dict__ = pickle.load(file)
            log.info('deprecated_model loaded , size: {} \t table_size: {} \t down_sampling: {} \t communities {}'.format(model.layer1_size,
                                                                                                               model.table_size,
                                                                                                               model.downsampling,
                                                                                                               model.k))
            return model
