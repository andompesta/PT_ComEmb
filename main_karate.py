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
    Helper function used to optimize O1
    :param network: neural network to train
    :param lr: learning rate
    :param model: model containing the shared data
    :param edges: numpy list of edges used for training
    :param num_iter: iteration number over the edges
    :param batch_size: size of the batch
    :return: loss value
    """
    log.info("computing o1")
    optimizer = SGD(network.parameters(), lr)

    num_batch = 0
    total_batch = (edges.shape[0] * num_iter) / batch_size
    loss_val = 0
    for batch in emb_utils.batch_generator(
            emb_utils.prepare_sentences(model,
                                        edges,
                                        network.transfer_fn(model.vocab)),
            batch_size):
        input, output = batch
        loss = network.forward(input, output, negative_sampling_fn=model.negative_sample)

        loss_val += loss.data[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        num_batch += 1

        if (num_batch) % 10000 == 0:
            log.info("community embedding batches completed: {}".format(num_batch/total_batch))

    log.debug("O1 loss: {}".format(loss_val))
    return loss_val

def learn_second(network, lr, model, examples_files, total_example, alpha=1.0, batch_size=20):
    """
    Helper function used to optimize O2
    :param network: network model to optimize
    :param lr: learning rate
    :param model: model containing the shared data
    :param examples_files: list of files containing the examples
    :param total_example: total example for training
    :param alpha: trade-off param
    :param batch_size: size of the batch
    :return: loss value
    """

    num_batch = 0

    log.info("compute o2")
    optimizer = SGD(network.parameters(), lr)
    log.debug("read example file: {}".format("\t".join(examples_files)))
    loss_val = 0

    if alpha <= 0:
        return loss_val

    for batch in emb_utils.batch_generator(
            emb_utils.prepare_sentences(model,
                                        graph_utils.combine_example_files_iter(examples_files),
                                        network.transfer_fn(model.vocab)),
            batch_size):
        input, output = batch
        loss = (alpha * network.forward(input, output, negative_sampling_fn=model.negative_sample))
        loss_val += loss.data[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        num_batch += 1

        if (num_batch) % 10000 == 0:
            log.info("community embedding batches completed: {}".format(num_batch/(total_example/batch_size)))

    log.debug("O2 loss: {}".format(loss_val))
    return loss_val


def learn_community(network, lr, model, nodes, num_iter=1, beta=1.0, batch_size=20):
    """
    Helper function used to optimize O3
    :param network: model to optimize
    :param lr: learning rate
    :param model: model containing the shared data
    :param nodes: nodes on which execute the learning
    :param num_iter: iteration number over the nodes
    :param beta: trade-off value
    :param batch_size: size of the batch
    :return: loss value
    """

    num_batch = 0

    log.info("compute o3")
    optimizer = SGD(network.parameters(), lr)
    loss_val = 0

    if beta <= 0.:
        return loss_val

    for batch in emb_utils.batch_generator(
            emb_utils.prepare_sentences(model,
                                        nodes,
                                        network.transfer_fn()),
            batch_size):

        input, output = batch
        loss = network.forward(input, model)
        loss.data *= (beta/model.k)
        loss_val += loss.data[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        num_batch += 1

        if (num_batch) % 10000 == 0:
            log.info("community embedding batches completed: {}".format(num_batch/(total_example/batch_size)))

    log.debug("O3 loss: {}".format(loss_val))
    return loss_val

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
    batch_size = 20
    input_file = "karate"
    output_file = "karate_my"




    G = graph_utils.load_adjacencylist('./data/' + input_file + '/' + input_file + '.adjlist', True)
    model = ComEModel(G.degree(),
                      size=representation_size,
                      input_file=input_file + '/' + input_file,
                      path_labels="./data")


    total_example = (G.number_of_nodes() * walk_length * num_walks * (2 * (window_size - 1)))
    num_iter = int(total_example/(G.number_of_nodes() * 2))

    # neg_loss = Context2Emb(deprecated_model, negative)
    o1_loss = Node2Emb(model, negative)
    o2_loss = Context2Emb(model, negative)
    o3_loss = Community2Emb(model, reg_covar=0.00001)

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

    io_utils.save_embedding(model.get_node_embedding(), "pytorch_embedding_random",
                  path="./data")

    # pre-training phase
    learn_second(o2_loss, lr, model, examples_files, total_example=total_example, alpha=alpha)
    learn_first(o1_loss, lr, model, edges, num_iter=num_iter)

    io_utils.save_embedding(model.get_node_embedding(), "pytorch_embedding_pre-train",
                  path="./data")

    assert np.array_equal(o1_loss.input_embeddings(), o2_loss.input_embeddings()), "node embedding is not consistent"
    assert np.array_equal(model.get_node_embedding(), o1_loss.input_embeddings()), "node embedding is not consistent"

    for it in range(1):
        o3_loss.fit(model)
        learn_first(o1_loss, lr, model, edges, num_iter=num_iter)
        learn_community(o3_loss, lr, model, zip(G.nodes_iter(), np.ones(G.number_of_nodes())),
                        beta=beta,
                        batch_size=batch_size)
        learn_second(o2_loss, lr, model, examples_files,
                     total_example=total_example,
                     alpha=alpha)



        io_utils.save_embedding(model.get_node_embedding(), "pytorch_embedding_ws-{}_rs-{}_alpha-{}_lr-{}_iter-{}".format(window_size,
                                                                                                     representation_size,
                                                                                                     alpha,
                                                                                                     lr,
                                                                                                     it),
                      path="./data")
