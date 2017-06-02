__author__ = 'ando'
import logging as log
import numpy as np
import pickle
from os.path import exists
from os import makedirs
from utils.embedding import Vocab
from utils.IO_utils import load_ground_true
from sklearn.base import TransformerMixin

log.basicConfig(format='%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s', level=log.DEBUG)


class ComEModel(object):
    '''
    class that keep track of all the parameters used during the learning of the embedding.
    '''


    def __init__(self, nodes_degree,
                 size=2,
                 down_sampling=0,
                 seed=1,
                 table_size=100000000,
                 node_embedding=None,
                 context_embedding=None,
                 path_labels='data/',
                 input_file=None):
        '''
        :param nodes_degree: Dict with node_id: degree of node
        :param size: projection space
        :param min_count: ignore all nodes with total frequency lower than this.
        :param downsampling: perform downsampling of common node
        :param seed: seed for random function
        :param table_size: size of the negative sampling table
        :param vocab: dictionary between a node and its count in the paths
        :param index2node: index between a node and its representation

        :param node_embedding: matrix containing the node embedding
        :param context_embedding: matrix containing the context embedding
        :param community_embedding: matrix containing the community embedding
        :param inv_covariance_mat: matrix representing the covariance matrix of the mixture clustering
        :param pi: probability distribution of each node respect the communities

        :param path_labels: location of the file containing the ground true (label for each node)
        :param input_file: name of the file containing the ground true (label for each node)
        :return:
        '''
        self.down_sampling = down_sampling
        self.seed = seed
        self.loss = 0

        if size % 4 != 0:
            log.warn("consider setting layer size to a multiple of 4 for greater performance")
        self.layer1_size = int(size)
        self.table_size = table_size

        if context_embedding is not None:
            self.context_embedding = context_embedding
        if node_embedding is not None:
            self.node_embedding = node_embedding
        
        if nodes_degree is not None:
            self.build_vocab_(nodes_degree)
            self.ground_true, self.k = load_ground_true(path=path_labels, file_name=input_file)
            # inizialize node and context embeddings
            self.reset_weights()
            self.make_table()
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

        self.node_embedding = np.random.uniform(-1., 1., (len(self.vocab), self.layer1_size)).astype(np.float32)
        self.context_embedding = np.zeros((len(self.vocab), self.layer1_size), dtype=np.float32)
        self.centroid = np.zeros((self.k, self.layer1_size), dtype=np.float32)
        self.covariance_mat = np.zeros((self.k, self.layer1_size, self.layer1_size), dtype=np.float32)
        self.inv_covariance_mat = np.zeros((self.k, self.layer1_size, self.layer1_size), dtype=np.float32)
        self.pi = np.zeros((len(self.vocab), self.k), dtype=np.float32)




    def make_table(self, power=0.75):
        """
        Create a table using stored vocabulary node counts for drawing random nodes in the negative
        sampling training routines.

        Called internally from `build_vocab()`.

        """
        log.info("constructing a table with noise distribution from %i nodes" % len(self.vocab))
        # table (= list of nodes) of noise distribution for negative sampling
        vocab_size = len(self.vocab)
        self.table = np.zeros(self.table_size, dtype=np.uint32)

        if not vocab_size:
            log.error("empty vocabulary in, is this intended?")
            return

        # compute sum of all power (Z in paper)
        train_nodes_pow = float(sum([self.vocab[node].count**power for node in self.vocab]))
        # go through the whole table and fill it up with the node indexes proportional to a node's count**power
        node_id = 1
        # normalize count^0.75 by Z
        d1 = self.vocab[node_id].count**power / train_nodes_pow
        log.debug("-----------Check if it has to be sorted by count or node_id--------")
        for tidx in range(self.table_size):
            self.table[tidx] = node_id
            if 1.0 * tidx / self.table_size > d1:
                node_id += 1
                d1 += self.vocab[node_id].count**power / train_nodes_pow
            if node_id >= vocab_size:
                node_id = vocab_size - 1
        log.info('max negative sample table %d' % max(self.table))



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
