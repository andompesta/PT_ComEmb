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


def learn_first(network, lr, model, edges, num_iter=1):
    """
    Helper function used to optimize O1 and O3
    :param network: neural network to train
    :param lr: learning rate
    :param model: deprecated_model used to compute the batches and the negative sampling
    :param edges: numpy list of edges used for training
    :param num_iter: iteration number over the edges
    :return: 
    """
    log.info("computing o1")
    optimizer = SGD(network.parameters(), lr)
    for batch in emb_utils.batch_generator(
            emb_utils.prepare_sentences(model,
                                        emb_utils.RepeatCorpusNTimes(edges, n=num_iter),
                                        network.transfer_fn(model.vocab)),
            20):
        input, output = batch
        loss = network.forward(input, output, negative_sampling_fn=model.negative_sample)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def learn_second(network, lr, model, examples_files, alpha=1.0):
    """
    Helper function used to optimize O1 and O3
    :param loss: loss to optimize
    :param lr: learning rate
    :param model: deprecated_model used to compute the batches and the negative sampling
    :param examples_files: list of files containing the examples
    :param num_iter: iteration number over the edges
    :return: 
    """
    log.info("compute o2")
    optimizer = SGD(network.parameters(), lr)
    log.debug("read example file: {}".format("\t".join(examples_files)))
    for batch in emb_utils.batch_generator(
            emb_utils.prepare_sentences(model,
                                        graph_utils.combine_example_files_iter(examples_files),
                                        network.transfer_fn(model.vocab)),
            20):
        input, output = batch
        loss = (alpha * network.forward(input, output, negative_sampling_fn=model.negative_sample))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


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
    path = "../data"

    G = graph_utils.load_adjacencylist(os.path.join(path, input_file, input_file + '.adjlist'), True)
    model = ComEModel(G.degree(),
                      size=representation_size,
                      input_file=input_file + '/' + input_file,
                      path_labels=path)
    model.k = 2
    o2_loss = Context2Emb(model, negative)
    o3_loss = Community2Emb(model, reg_covar=0.00001)

    node_color = plot_utils.graph_plot(G=G,
                                       show=False,
                                       graph_name="karate",
                                       node_position_file=True,
                                       node_position_path=path)

    exmple_filebase = os.path.join(path, output_file + ".exmple")  # where read/write the sampled path
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

    learn_second(o2_loss, lr, model, examples_files, alpha=alpha)
    node_embeddings = o2_loss.input_embeddings()
    io_utils.save(node_embeddings, "pytorch_embedding_test_o2", path="../data")

    assert np.array_equal(model.get_node_embedding(), node_embeddings)
    # test o3
    o3_loss.fit(model)
    optimizer = SGD(o3_loss.parameters(), lr)

    loss = o3_loss.forward(model, beta)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    io_utils.save(model.get_node_embedding(), "pytorch_embedding_test_o2_o3-", path="../data")
