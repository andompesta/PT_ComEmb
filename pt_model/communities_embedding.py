__author__ = 'ando'
import sklearn.mixture as mixture
import numpy as np
import logging as log
from scipy.stats import multivariate_normal
import torch.nn as nn
import torch as t
from torch.autograd import Function

log.basicConfig(format='%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s', level=log.DEBUG)

short_dtype = t.cuda.ShortTensor
float_tensor = t.cuda.FloatTensor


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
    def forward(self, input, model, lambda2=0):
        """
        Forward function used to compute o3 loss
        :param input: node_embedding of the batch
        :param model: model information from the GMM
        :param lambda2: factor to modeulate the loss
        """
        ret_loss = []
        for com in range(model.k):
            rd = multivariate_normal(model.centroid[com], model.covariance_mat[com])
            # check if can be done as matrix operation
            ret_loss.append(rd.logpdf(input) * model.pi[:, com])
        ret_loss = sum(ret_loss)
        self.save_for_backward(input)
        return ret_loss * (-lambda2/model.k)

    def backward(self, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input
        :param grad_output: loss
        :return:
        """
        #TODO: it is not a neural net. Has the gradient to depend on the loss?
        # check http://pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-defining-new-autograd-functions to bette runderstand autograd

        return 0

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

    def fit(self, input, model):
        '''
        Fit the GMM model with the current node embedding and save the result in the model
        '''
        log.debug("num community: {}".format(model.k))
        self.g_mixture.fit(input)

        # diag_covars = []
        # for covar in g.covariances_:
        #     diag = np.diag(covar)
        #     diag_covars.append(diag)

        model.centroid = float_tensor(self.g_mixture.means_)
        model.covariance_mat = float_tensor(self.g_mixture.covariances_)
        model.inv_covariance_mat = t.inverse(model.covariance_mat)
        model.pi = float_tensor(self.g_mixture.predict_proba(model.node_embedding))


    def forward(self, model, lambda2):
        return Community2EmbFn()(self.node_embedding, model, lambda2)
