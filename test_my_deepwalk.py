__author__ = 'ando'

import logging as log
import configparser
import os
import random
from multiprocessing import cpu_count
import numpy as np
import psutil
from model.my_word2vec import ComEModel
from model.context_embedding import Context2Vec
import sys
import utils.IO_utils as io_utils
import utils.graph_utils as graph_utils
import utils.plot_utils as plot_utils

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

    input_file = "karate"
    output_file = "karate_my"

    G = graph_utils.load_adjacencylist('data/' + input_file + '/' + input_file + '.adjlist', True)
    node_color = plot_utils.graph_plot(G=G,
                                       show=True,
                                       graph_name="karate",
                                       node_position_file=True,
                                       node_position_path='data')

    walks_filebase = os.path.join("./data/", output_file + ".walks")                       # where read/write the sampled path

    # Sampling the random walks for context
    walk_files = None
    log.info("sampling the paths")
    walk_files = graph_utils.write_walks_to_disk(G, walks_filebase,
                                                 num_paths=num_walks,
                                                 path_length=walk_length,
                                                 alpha=0,
                                                 rand=random.Random(9999999999),
                                                 num_workers=num_workers)



    #Learning algorithm
    cont_learner = Context2Vec(window_size=window_size, workers=num_workers, negative=negative, alpha=0.025)
    vertex_counts = graph_utils.count_textfiles(walk_files, num_workers)

    model = ComEModel(vertex_counts,
                      size=representation_size,
                      table_size=1000000,
                      input_file=input_file + '/' + input_file)


    context_total_nodes = G.number_of_nodes() * num_walks * walk_length
    log.debug("context_total_nodes: %d" % (context_total_nodes))

    cont_learner.train(model, graph_utils.combine_files_iter(walk_files), context_total_nodes)
    io_utils.save_embedding(model.node_embedding, file_name="{}_ComE_l1-{}_l2-{}_ds-{}_it-{}".format(output_file,
                                                                                                     0,
                                                                                                     0,
                                                                                                     0,
                                                                                                     0))
