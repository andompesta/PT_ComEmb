__author__ = 'ando'

import logging as log
import time
import threading
from queue import Queue
import numpy as np
from utils.embedding import chunkize_serial, prepare_sentences
from scipy.special import expit as sigmoid

log.basicConfig(format='%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s', level=log.DEBUG)

class Context2Vec(object):
    '''
    Class that train the context embedding
    '''
    def __init__(self, alpha=0.1, window_size=5, workers=1, negative=5):
        '''
        :param alpha: learning rate
        :param window: windows size used to compute the context embeddings
        :param workers: number of thread
        :param negative: number of negative samples
        :return:
        '''

        self.alpha = float(alpha)
        self.workers = workers
        self.negative = negative
        self.window_size = int(window_size)

    def train(self, model, paths, total_nodes, chunksize=150):
        """
        Update the model's neural weights from a sequence of paths (can be a once-only generator stream).
        """
        assert model.node_embedding.dtype == np.float32
        assert model.context_embedding.dtype == np.float32


        log.info("training model with %i workers on %i vocabulary and %i features and 'negative sampling'=%s" %
                    (self.workers, len(model.vocab), model.layer1_size, self.negative))

        if not model.vocab:
            raise RuntimeError("you must first build vocabulary before training the model")

        start, next_report = time.time(), [1.0]
        if total_nodes is None:
            raise AttributeError('need to the the number of node')

        node_count = [0]
        jobs = Queue(maxsize=2*self.workers)  # buffer ahead only a limited number of jobs.. this is the reason we can't simply use ThreadPool :(
        lock = threading.Lock()  # for shared state (=number of nodes trained so far, log reports...)

        def worker_train():
            """Train the model, lifting lists of paths from the jobs queue."""
            while True:
                job = jobs.get()
                if job is None:  # data finished, exit
                    break

                # update the learning rate before every job
                # alpha = max(self.min_alpha, self.alpha * (1 - 1.0 * node_count[0] / total_nodes))
                # how many nodes did we train on? out-of-vocabulary (unknown) nodes do not count

                job_nodes = sum(self.train_sg(model.node_embedding,
                                              model.context_embedding,
                                              path,
                                              self.alpha,
                                              self.negative,
                                              self.window_size,
                                              model.table) for path in job) #execute the sgd

                with lock:
                    node_count[0] += job_nodes
                    # loss[0] += job_loss

                    elapsed = time.time() - start
                    if elapsed >= next_report[0]:
                        log.info("PROGRESS: at %.2f%% nodes, alpha %.05f, %.0f nodes/s" %
                                    (100.0 * node_count[0] / total_nodes, self.alpha, node_count[0] / elapsed if elapsed else 0.0))
                        next_report[0] = elapsed + 1.0  # don't flood the log, wait at least a second between progress reports

        workers = [threading.Thread(target=worker_train) for _ in range(self.workers)]
        for thread in workers:
            thread.daemon = True  # make interrupting the process with ctrl+c easier
            thread.start()

        # convert input strings to Vocab objects (eliding OOV/downsampled nodes), and start filling the jobs queue
        for job_no, job in enumerate(chunkize_serial(prepare_sentences(model, paths), chunksize)):
            jobs.put(job)

        log.info("reached the end of input; waiting to finish %i outstanding jobs" % jobs.qsize())
        for _ in range(self.workers):
            jobs.put(None)  # give the workers heads up that they can finish -- no more work!

        for thread in workers:
            thread.join()

        elapsed = time.time() - start
        log.info("training on %i nodes took %.1fs, %.0f nodes/s" %
                    (node_count[0], elapsed, node_count[0] / elapsed if elapsed else 0.0))


    def train_sg(self, py_node_embedding, py_context_embedding, py_path, py_alpha, py_negative, py_window, py_table,
                 py_work=None):

        """
        Update skip-gram model by training on a single path.

        The path is a list of Vocab objects (or None, where the corresponding
        node is not in the vocabulary. Called internally from `node2Vec.train()`.

        This is the non-optimized, Python version. If you have cython installed, gensim
        will use the optimized version from node2vec_inner instead.

        """

        # sdg gradient update
        def gradient_update(positive_node_embedding, negative_context_embedding, neg_labels, alpha):
            fb = sigmoid(np.dot(positive_node_embedding, negative_context_embedding.T))  # propagate hidden -> output
            gb = (neg_labels - fb) * alpha  # vector of error gradients multiplied by the learning rate
            return gb

        result = 0
        for pos, node in enumerate(py_path):  # node = input vertex of the sistem
            if node is None:
                continue  # OOV node in the input path => skip

            labels = np.zeros(py_negative + 1)
            labels[0] = 1.0  # frist node come from the path, the other not (lable[1:]=0)

            start = max(0, pos - py_window)
            # now go over all nodes from the (reduced) window, predicting each one in turn
            for pos2, node2 in enumerate(py_path[start: pos + py_window + 1], start):  # node 2 are the output nodes predicted form node


                # don't train on OOV nodes and on the `node` itself
                if node2 and not (pos2 == pos):
                    positive_node_embedding = py_node_embedding[node2.index]  # correct node embeddings
                    py_work = np.zeros(positive_node_embedding.shape)  # reset the error for each new sempled node
                    '''
                    perform negative sampling
                    '''
                    # use this node (label = 1) + `negative` other random nodes not from this path (label = 0)
                    node_indices = [node.index]
                    while len(node_indices) < py_negative + 1:
                        w = py_table[np.random.randint(py_table.shape[0])]  # sample a new negative node
                        if w != node.index and w != node2.index:
                            node_indices.append(w)

                    # node/context embedding sgd
                    negative_nodes_embedding = py_context_embedding[node_indices]
                    update_sgd = gradient_update(positive_node_embedding, negative_nodes_embedding, labels, py_alpha)

                    py_context_embedding[node_indices] += np.outer(update_sgd,
                                                                   positive_node_embedding)  # Update context embeddings
                    py_work += np.dot(update_sgd, negative_nodes_embedding)


                    # SDG 2
                    # py_sgd2 = sgd2(py_node_embedding, py_centroid, py_inv_covariance_mat, py_pi, py_k, py_alpha, py_lambda2, node2.index )
                    py_node_embedding[node2.index] += py_work # Update node embeddings
        result += len(py_path)
        return result