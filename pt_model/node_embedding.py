__author__ = 'ando'

import logging as log
from functools import partial
from torch.autograd import Variable
import torch as t
import torch.nn as nn

log.basicConfig(format='%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s', level=log.DEBUG)



class Node2Emb(nn.Module):
    '''
    Class that train the context embedding
    '''
    def __init__(self, model, negative=5):
        '''
        :param alpha: learning rate
        :param window: windows size used to compute the context embeddings
        :param workers: number of thread
        :param negative: number of negative samples
        :return:
        '''
        super(Node2Emb, self).__init__()
        self.negative = negative
        self.node_embedding = model.node_embedding
        self.context_embedding = model.node_embedding

    def forward(self, input_labels, out_labels, negative_sampling_fn):
        """
        :param input_labes: Tensor with shape of [batch_size] of Long type
        :param out_labels: Tensor with shape of [batch_size, window_size] of Long type
        :param negative_sampling_fn: Function that sample negative nodes based on the weights
        :return: Loss estimation with shape of [1]
        """
        use_cuda = self.context_embedding.weight.is_cuda

        [batch_size] = out_labels.size()

        input = self.node_embedding(Variable(input_labels.view(-1)))
        output = self.context_embedding(Variable(out_labels.view(-1)))

        log_target = (input * output).sum(1).squeeze().sigmoid().log()  # loss of the positive example
        # equivalente to t.bmm(t.transpose(t.unsqueeze(input, 2),1, 2), t.unsqueeze(output, 2)).squeeze()

        # SUBSAMPLE
        noise_sample_count = batch_size * self.negative
        draw = negative_sampling_fn(noise_sample_count)
        draw.resize((batch_size, self.negative))
        noise = Variable(t.from_numpy(draw))

        if use_cuda:
            noise = noise.cuda()
        noise = self.context_embedding(noise).neg()
        sum_log_sampled = t.bmm(noise, input.unsqueeze(2)).sigmoid().log().sum(1).squeeze()


        loss = log_target + sum_log_sampled

        return -loss.sum() / batch_size

    def input_embeddings(self):
        return self.node_embedding.weight.data.cpu().numpy()

    def transfer_fn(self, model_dic):
        '''
        Transfer function form edge=[node_id_1 node_id_b] to edge=[node_idx_1 node_idx_b]
        :param model_dic: dictionary form node_id to node_data. Used for index substitution and sampling
        '''
        return partial(lambda x, vocab: vocab[x].index, vocab=model_dic)
