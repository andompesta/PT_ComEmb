__author__ = 'ando'

import logging as log
import time
import numpy as np
from torch.autograd import Variable
from utils.embedding import chunkize_serial, prepare_sentences
from scipy.special import expit as sigmoid
from collections import deque
import torch as t
import torch.nn as nn
import random

log.basicConfig(format='%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s', level=log.DEBUG)
short_dtype = t.cuda.ShortTensor

class Context2Vec(nn.Module):
    '''
    Class that train the context embedding
    '''
    def __init__(self, alpha=0.1, window_size=5, negative=5):
        '''
        :param alpha: learning rate
        :param window: windows size used to compute the context embeddings
        :param workers: number of thread
        :param negative: number of negative samples
        :return:
        '''
        super(Context2Vec, self).__init__()
        self.alpha = float(alpha)
        self.negative = negative
        self.window_size = int(window_size)

    def forward(self, input_labels, out_labels, num_sampled, model):
        """
        :param input_labes: Tensor with shape of [batch_size] of Long type
        :param out_labels: Tensor with shape of [batch_size, window_size] of Long type
        :param num_sampled: An int. The number of sampled from noise examples
        :return: Loss estimation with shape of [1]
        """

        use_cuda = model.context_embedding.weight.is_cuda

        [batch_size, window_size] = out_labels.size()

        input = model.node_embedding(input_labels.repeat(1, window_size).contiguous().view(-1))
        output = model.context_embedding(out_labels.view(-1))

        if model.sampling_weight is not None:
            # SUBSAMPLE
            noise_sample_count = batch_size * window_size * num_sampled
            draw = model.negative_sample(noise_sample_count)
            draw.resize((batch_size * window_size, num_sampled))
            noise = Variable(t.from_numpy(draw))
        else:
            # UNIFORMLY DISTRIBUTED SAMPLING
            noise = Variable(t.Tensor(batch_size * window_size,
                                      num_sampled).uniform_(0, len(model.vocab) - 1).long())

        if use_cuda:
            noise = noise.cuda()
        noise = model.context_embedding(noise).neg()

        log_target = (input * output).sum(1).squeeze().sigmoid().log()          # loss of the positive example

        ''' ∑[batch_size * window_size, num_sampled, embed_size] * [batch_size * window_size, embed_size, 1] ->
            ∑[batch_size, num_sampled, 1] -> [batch_size] '''
        sum_log_sampled = t.bmm(noise, input.unsqueeze(2)).sigmoid().log().sum(1).squeeze()     # loss of the negative example

        loss = log_target + sum_log_sampled

        return -loss.sum() / batch_size