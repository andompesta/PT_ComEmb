__author__ = 'ando'

import logging as log
import os
import random
from multiprocessing import cpu_count
import numpy as np
import psutil
from pt_model.model import ComEModel
from pt_model.context_embedding import Context2Emb
from pt_model.node_embedding import Node2Emb
from pt_model.communities_embedding import Community2Emb

import sys
import utils.IO_utils as io_utils
import utils.graph_utils as graph_utils
import utils.plot_utils as plot_utils
import utils.embedding as emb_utils
from torch.optim.sgd import SGD



p = psutil.Process(os.getpid())
try:
    p.set_cpu_affinity(list(range(cpu_count())))
except AttributeError:
    try:
        p.cpu_affinity(list(range(cpu_count())))
    except AttributeError:
        pass
log.basicConfig(format='%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s', level=log.DEBUG)
def debug(type_, value, tb):
    if hasattr(sys, 'ps1') or not sys.stderr.isatty():
        sys.__excepthook__(type_, value, tb)
    else:
        import traceback
        import pdb
        traceback.print_exception(type_, value, tb)
        print(u"\n")
        pdb.pm()


def learn_first(network, lr, model, edges, num_iter=1, batch_size=20):
    """
    Helper function used to optimize O1 and O3
    :param network: neural network to train
    :param lr: learning rate
    :param model: deprecated_model used to compute the batches and the negative sampling
    :param edges: numpy list of edges used for training
    :param num_iter: iteration number over the edges
    :param batch_size: size of the batch
    :return: loss value
    """
    log.info("computing o1")
    optimizer = SGD(network.parameters(), lr)

    num_batch = 0
    total_batch = (edges.shape[0] * num_iter) / batch_size

    for batch in emb_utils.batch_generator(
            emb_utils.prepare_sentences(model,
                                        emb_utils.RepeatCorpusNTimes(edges, n=num_iter),
                                        network.transfer_fn(model.vocab)),
            batch_size):
        input, output = batch
        loss = network.forward(input, output, negative_sampling_fn=model.negative_sample)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        num_batch += 1

        if (num_batch) % 10000 == 0:
            log.info("community embedding batches completed: {}".format(num_batch/total_batch))

    return loss

def learn_second(network, lr, model, examples_files, total_example, alpha=1.0, batch_size=20):
    """
    Helper function used to optimize O1 and O3
    :param loss: loss to optimize
    :param lr: learning rate
    :param model: deprecated_model used to compute the batches and the negative sampling
    :param examples_files: list of files containing the examples
    :param num_iter: iteration number over the edges
    :return: loss value
    """

    num_batch = 0

    log.info("compute o2")
    optimizer = SGD(network.parameters(), lr)
    log.debug("read example file: {}".format("\t".join(examples_files)))
    for batch in emb_utils.batch_generator(
            emb_utils.prepare_sentences(model,
                                        graph_utils.combine_example_files_iter(examples_files),
                                        network.transfer_fn(model.vocab)),
            batch_size):
        input, output = batch
        loss = (alpha * network.forward(input, output, negative_sampling_fn=model.negative_sample))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        num_batch += 1

        if (num_batch) % 10000 == 0:
            log.info("community embedding batches completed: {}".format(num_batch/(total_example/batch_size)))

    return loss

if __name__ == "__main__":
    num_walks = 10
    walk_length = 20
    window_size = 3
    negative = 4
    representation_size = 2
    num_workers = 1
    lr = 0.1
    alpha = 1.0
    beta = 1.0
    num_iter = 200
    input_file = "karate"
    output_file = "karate_my"

    G = graph_utils.load_adjacencylist('./data/' + input_file + '/' + input_file + '.adjlist', True)
    model = ComEModel(G.degree(),
                      size=representation_size,
                      input_file=input_file + '/' + input_file,
                      path_labels="./data")

    # neg_loss = Context2Emb(deprecated_model, negative)
    o1_loss = Node2Emb(model, negative)
    o2_loss = Context2Emb(model, negative)
    o3_loss = ComEModel

    node_color = plot_utils.graph_plot(G=G,
                                       show=False,
                                       graph_name="karate",
                                       node_position_file=True,
                                       node_position_path='./data')

    exmple_filebase = os.path.join("./data/", output_file + ".exmple")                       # where read/write the sampled path
    num_iter = G.number_of_nodes() * num_walks * walk_length

    # Sampling the random walks for context
    log.info("sampling the paths")
    examples_files = graph_utils.write_walks_to_disk(G, exmple_filebase,
                                                 windows_size=window_size,
                                                 num_paths=num_walks,
                                                 path_length=walk_length,
                                                 alpha=0,
                                                 rand=random.Random(9999999999),
                                                 num_workers=num_workers)
    edges = np.array(G.edges())
    edges = np.concatenate((edges, np.fliplr(edges)))

    learn_first(o1_loss, lr, model, edges, num_iter=num_iter)
    learn_second(o2_loss, lr, model, examples_files, total_example=(G.number_of_nodes() * walk_length * num_walks * (2*(window_size-1))),
                 alpha=alpha)

    assert np.array_equal(o1_loss.input_embeddings(), o2_loss.input_embeddings()), "node embedding is not the same"
    node_embeddings = o1_loss.input_embeddings()

    # test o3




    io_utils.save(node_embeddings, "pytorch_embedding_ws-{}_rs-{}_alpha-{}_lr-{}_iter-{}".format(window_size,
                                                                                                 representation_size,
                                                                                                 alpha,
                                                                                                 lr,
                                                                                                 num_iter),
                  path="./data")
