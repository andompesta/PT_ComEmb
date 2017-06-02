__author__ = 'ando'

import logging as log
from torch.autograd import Variable
import torch as t
import torch.nn as nn
from functools import partial

log.basicConfig(format='%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s', level=log.DEBUG)

class Context2Emb(nn.Module):
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
        super(Context2Emb, self).__init__()
        self.negative = negative
        self.node_embedding = model.node_embedding
        self.context_embedding = model.context_embedding
        self.weight = model.sampling_weight

    def forward(self, input_labels, out_labels, negative_sampling_fn):
        """
        :param input_labes: Tensor with shape of [batch_size] of Long type
        :param out_labels: Tensor with shape of [batch_size, window_size] of Long type
        :param negative_sampling_fn: Function that sample negative nodes based on the weights
        :return: Loss estimation with shape of [1]
        """

        use_cuda = self.context_embedding.weight.is_cuda

        [batch_size, window_size] = out_labels.size()

        input = self.node_embedding(Variable(input_labels.view(batch_size, 1).repeat(1, window_size).view(-1)))
        output = self.context_embedding(Variable(out_labels.view(-1)))


        # SUBSAMPLE
        noise_sample_count = batch_size * window_size * self.negative
        draw = negative_sampling_fn(noise_sample_count)
        draw.resize((batch_size * window_size, self.negative))
        noise = Variable(t.from_numpy(draw))

        if use_cuda:
            noise = noise.cuda()
        noise = self.context_embedding(noise).neg()

        log_target = (input * output).sum(1).squeeze().sigmoid().log()          # loss of the positive example

        ''' ∑[batch_size * window_size, num_sampled, embed_size] * [batch_size * window_size, embed_size, 1] ->
            ∑[batch_size, num_sampled, 1] -> [batch_size] '''
        sum_log_sampled = t.bmm(noise, input.unsqueeze(2)).sigmoid().log().sum(1).squeeze()     # loss of the negative example

        loss = log_target + sum_log_sampled

        return -loss.sum() / batch_size

    def input_embeddings(self):
        return self.node_embedding.weight.data.cpu().numpy()

    def transfer_fn(self, model_dic):
        """
        Transfer function from path=[node_id_1, node_id_2, ...] to path=[node_idx_1, node_idx_2, ...]
        :param model_dic: dictionary form node_id to node_data. Used for index substitution and sampling
        """
        return lambda input: list(map(partial(lambda x, vocab: vocab[x].index, vocab=model_dic), input))
