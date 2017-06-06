__author__ = 'ando'
import sklearn.mixture as mixture
import numpy as np
import logging as log
from functools import partial
from scipy.stats import multivariate_normal
import torch.nn as nn
import torch as t
from torch.autograd import Function, Variable

log.basicConfig(format='%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s', level=log.DEBUG)
float_tensor = None


class Community2EmbFn(Function):
    def __init__(self, model, input_idx):
        super(Community2EmbFn, self).__init__()
        self.centroid = model.centroid
        self.pi = model.pi
        self.covariance_mat = model.covariance_mat
        self.inv_covariance_mat = model.inv_covariance_mat
        self.k = model.k
        self.input_idx = input_idx
        self.size = input_idx.shape[0]

    def forward(self, input):
        """
        Forward function used to compute o3 loss
        :param input: node_embedding of the batch
        """
        self.save_for_backward(input)
        input = input.cpu().numpy()

        ret_loss = np.zeros(self.size, dtype=np.float32)
        for com in range(self.k):
            rd = multivariate_normal(self.centroid[com], self.covariance_mat[com])
            # check if can be done as matrix operation
            ret_loss += rd.logpdf(input).astype(np.float32) * self.pi[self.input_idx, com]

        ret_loss = float_tensor(1).fill_(abs(float(ret_loss.sum())))

        return ret_loss

    def backward(self, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input
        :param grad_output: initial loss of the computation graph (=1)
        :return:
        """

        input, = self.saved_tensors
        # log.debug(input)
        # log.debug(input.cpu().numpy() - self.centroid[0])
        # log.debug(grad_output)
        grad_input = t.zeros(input.size()).type(float_tensor)

        for com in range(self.k):
            diff = input.cpu().numpy() - self.centroid[com]
            m = self.pi[self.input_idx, com].reshape(self.size, 1, 1) * self.inv_covariance_mat[com]
            diff = float_tensor(diff).unsqueeze_(-1)
            grad_input += t.bmm(float_tensor(m), diff).squeeze()
            # log.debug("m: {}".format(m))
            # log.debug("grad: {}".format(grad_input))

        grad_input.clamp_(min=-0.2, max=0.2)
        # log.debug(grad_input)
        return grad_input

class Community2Emb(nn.Module):
    '''scipy based GMM/BGMM model'''
    def __init__(self, model, reg_covar=0, f_type=t.FloatTensor):
        '''
        :param window: windows size used to compute the context embeddings
        :return:
        '''
        super(Community2Emb, self).__init__()
        self.node_embedding = model.node_embedding
        self.g_mixture = mixture.GaussianMixture(n_components=model.k, reg_covar=reg_covar, covariance_type='full', n_init=10)
        global float_tensor
        float_tensor = f_type

    def get_node_embedding(self):
        return self.node_embedding.weight.data.cpu().numpy()

    def fit(self, model):
        '''
        Fit the GMM model with the current node embedding and save the result in the model
        :param model: model injected to add the mixture parameters
        '''

        node_embedding = self.get_node_embedding()
        log.info("Fitting: {} communities".format(model.k))
        self.g_mixture.fit(node_embedding)

        # diag_covars = []
        # for covar in g.covariances_:
        #     diag = np.diag(covar)
        #     diag_covars.append(diag)

        model.centroid = self.g_mixture.means_.astype(np.float32)
        model.covariance_mat = self.g_mixture.covariances_.astype(np.float32)
        model.inv_covariance_mat = np.linalg.inv(model.covariance_mat).astype(np.float32)
        model.pi = self.g_mixture.predict_proba(node_embedding).astype(np.float32)
        # model.score = self.g_mixture.score_samples(node_embedding)

    def forward(self, input_labels, model):
        input = self.node_embedding(Variable(input_labels))
        return Community2EmbFn(model, input_labels.cpu().numpy())(input)


    def transfer_fn(self):
        return lambda x: 0
