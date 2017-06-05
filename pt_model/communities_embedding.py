__author__ = 'ando'
import sklearn.mixture as mixture
import numpy as np
import logging as log
from scipy.stats import multivariate_normal
import torch.nn as nn
import torch as t
from torch.autograd import Function, Variable

log.basicConfig(format='%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s', level=log.DEBUG)

float_tensor = t.FloatTensor
long_tensor = t.LongTensor


def sdg(node_embedding, centroid, inv_covariance_mat, pi, k, _alpha, _lambda2, index):
    _sum = np.zeros(node_embedding[index].shape, dtype=np.float32)
    if _lambda2 > 0:
        for com in range(k):
            # modulation_factor = np.array([ elem * np.linalg.inv(model.covariance_mat[com]) for elem in np.nditer(model.pi[:, com])])
            # differentation = (model.node_embedding - model.community_embedding[com])
            # modulation_factor = np.outer(model.pi[:, com], np.linalg.inv(model.covariance_mat[com])).reshape(shape)
            # _sum += [(elems[0].dot(elems[1])) for elems in zip(modulation_factor, differentation)]
            diff = (node_embedding[index] - centroid[com])
            m = pi[index, com] * inv_covariance_mat[com]
            _sum += np.dot(m, diff) * _lambda2
    return - np.clip((_sum * _alpha), -0.05, 0.05)


class Community2EmbFn(Function):
    def __init__(self, model, beta=1.0):
        super(Community2EmbFn, self).__init__()
        self.centroid = model.centroid
        self.pi = model.pi
        self.covariance_mat = model.covariance_mat
        self.inv_covariance_mat = model.inv_covariance_mat
        self.k = model.k
        self.beta = beta
        self.size = model.size

    def forward(self, input):
        """
        Forward function used to compute o3 loss
        :param input: node_embedding of the batch
        """
        self.save_for_backward(input)

        if self.beta == 0:
            return 0.

        ret_loss = np.zeros(self.size)
        for com in range(self.k):
            rd = multivariate_normal(self.centroid[com], self.covariance_mat[com])
            # check if can be done as matrix operation
            ret_loss += rd.logpdf(input.cpu().numpy()) * self.pi[:, com]

        ret_loss = float_tensor([ret_loss.sum()])

        return ret_loss * (-self.beta/self.k)

    def backward(self, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input
        :param grad_output: loss
        :return:
        """
        #TODO: check if the invers of the covariance is needed

        input, = self.saved_tensors
        # log.debug(input)
        # log.debug(input.cpu().numpy() - self.centroid[0])
        # log.debug(grad_output)
        grad_input = t.zeros(input.size()).type(float_tensor)

        for com in range(self.k):
            diff = input.cpu().numpy() - self.centroid[com]
            m = self.pi[:, com].reshape(self.size, 1, 1) * self.inv_covariance_mat[com]
            diff = float_tensor(diff).unsqueeze_(-1)
            grad_input += t.bmm(float_tensor(m), diff).squeeze()
            # log.debug("m: {}".format(m))
            # log.debug("grad: {}".format(grad_input))

        grad_input.clamp_(min=-0.5, max=0.5)
        # log.debug(grad_input)
        return grad_input

class Community2Emb(nn.Module):
    '''scipy based GMM/BGMM model'''
    def __init__(self, model, reg_covar=0):
        '''
        :param window: windows size used to compute the context embeddings
        :return:
        '''
        super(Community2Emb, self).__init__()
        self.node_embedding = model.node_embedding
        self.g_mixture = mixture.GaussianMixture(n_components=model.k, reg_covar=reg_covar, covariance_type='full', n_init=5)

    def fit(self, model):
        '''
        Fit the GMM model with the current node embedding and save the result in the model
        '''
        node_embedding = self.node_embedding.weight.data.cpu().numpy()

        log.info("Fitting: {} communities".format(model.k))
        self.g_mixture.fit(node_embedding)

        # diag_covars = []
        # for covar in g.covariances_:
        #     diag = np.diag(covar)
        #     diag_covars.append(diag)

        model.centroid = self.g_mixture.means_
        model.covariance_mat = self.g_mixture.covariances_
        model.inv_covariance_mat = np.linalg.inv(model.covariance_mat)
        model.pi = self.g_mixture.predict_proba(node_embedding)
        # model.score = self.g_mixture.score_samples(node_embedding)

    def forward(self, model, beta):
        input_labels = long_tensor(range(model.size))
        input = self.node_embedding(Variable(input_labels))
        return Community2EmbFn(model, beta)(input)
