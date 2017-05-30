__author__ = 'ando'

import logging as log
import configparser
import os
import random
from multiprocessing import cpu_count
import numpy as np
import psutil
from pt_model.model import ComEModel
from pt_model.context_embedding import Context2Emb
from pt_model.node_embedding import Node2Emb
import sys
import utils.IO_utils as io_utils
import utils.graph_utils as graph_utils
import utils.plot_utils as plot_utils
import utils.embedding as emb_utils
from torch.optim.sgd import SGD
import timeit



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




if __name__ == "__main__":
    num_walks = 10
    walk_length = 20
    window_size = 3
    negative = 4
    representation_size = 2
    num_workers = 1

    lambda_1 = 1.0
    lambda_2 = 1.0

    input_file = "karate"
    output_file = "karate_my"

    G = graph_utils.load_adjacencylist('../data/' + input_file + '/' + input_file + '.adjlist', True)
    model = ComEModel(G.degree(),
                      size=representation_size,
                      input_file=input_file + '/' + input_file,
                      path_labels="../data")

    # neg_loss = Context2Emb(model, negative)
    neg_loss = Node2Emb(model, negative)
    optimizer = SGD(neg_loss.parameters(), 0.1)


    node_color = plot_utils.graph_plot(G=G,
                                       show=False,
                                       graph_name="karate",
                                       node_position_file=True,
                                       node_position_path='../data')

    exmple_filebase = os.path.join("../data/", output_file + ".exmple")                       # where read/write the sampled path

    # Sampling the random walks for context
    log.info("sampling the paths")
    example_files = graph_utils.write_walks_to_disk(G, exmple_filebase,
                                                 windows_size=window_size,
                                                 num_paths=num_walks,
                                                 path_length=walk_length,
                                                 alpha=0,
                                                 rand=random.Random(9999999999),
                                                 num_workers=num_workers)
    edges = np.array(G.edges())
    edges = np.concatenate((edges, np.fliplr(edges)))

    for batch in emb_utils.batch_generator(
            emb_utils.prepare_sentences(model,
                                        emb_utils.RepeatCorpusNTimes(edges, n=200),
                                        # graph_utils.combine_example_files_iter(example_files),
                                        neg_loss.transfer_fn(model.vocab)),
            20):
        input, output = batch
        loss = lambda_1 * neg_loss.forward(input, output, negative_sampling_fn=model.negative_sample)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    word_embeddings = neg_loss.input_embeddings()
    io_utils.save(word_embeddings, "pytorch_embedding_o1", path="../data")