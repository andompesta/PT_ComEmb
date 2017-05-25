#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging
import psutil
from multiprocessing import cpu_count

from deepwalk import walks as serialized_walks
from deepwalk.skipgram import Skipgram

from utils.plot_utils import graph_plot, node_space_plot_2D
import utils.graph_utils as graph_utils
import utils.IO_utils as io_utils
p = psutil.Process(os.getpid())
try:
    p.set_cpu_affinity(list(range(cpu_count())))
except AttributeError:
    try:
        p.cpu_affinity(list(range(cpu_count())))
    except AttributeError:
        pass

logger = logging.getLogger(__name__)
LOGFORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"


def debug(type_, value, tb):
    if hasattr(sys, 'ps1') or not sys.stderr.isatty():
        sys.__excepthook__(type_, value, tb)
    else:
        import traceback
        import pdb
        traceback.print_exception(type_, value, tb)
        print(u"\n")
        pdb.pm()


def process(args):
    G = graph_utils.load_adjacencylist(os.path.join("../data", args.input), undirected=args.undirected)
    num_walks = len(G.nodes()) * args.number_walks
    data_size = num_walks * args.walk_length

    print("Number of nodes: {}".format(len(G.nodes())))
    print("Number of walks: {}".format(num_walks))
    print("Data size (walks*length): {}".format(data_size))

    walks_filebase = args.output + ".walks"


    # print("Data size {} is larger than limit (max-memory-data-size: {})."
    #       "Dumping walks to disk.".format(data_size, args.max_memory_data_size))
    print("Walking...")
    walk_files = graph_utils.write_walks_to_disk(G, walks_filebase,
                                                 num_paths=args.number_walks,
                                                 path_length=args.walk_length,
                                                 alpha=0,
                                                 rand=random.Random(9999999999),
                                                 num_workers=args.workers)

    print("Counting vertex frequency...")
    if not args.vertex_freq_degree:
        vertex_counts = serialized_walks.count_textfiles(walk_files, args.workers)
    else:
        # use degree distribution for frequency in tree
        vertex_counts = G.degree(nodes=G.iterkeys())

    print("Training...")
    model = Skipgram(sentences=serialized_walks.combine_files_iter(walk_files),
                     vocabulary_counts=vertex_counts,
                     size=args.representation_size,
                     window=args.window_size,
                     min_count=0,
                     workers=args.workers)

    io_utils.save_embedding(model.syn0, file_name="{}".format(args.output), path="")
    model.save_word2vec_format("{}.emb".format(args.output))


def main():
    parser = ArgumentParser("deepwalk",
                          formatter_class=ArgumentDefaultsHelpFormatter,
                          conflict_handler='resolve')

    parser.add_argument("--debug", dest="debug", action='store_true', default=False,
                      help="drop a debugger if an exception is raised.")

    parser.add_argument('--format', default='adjlist',
                      help='File format of input file')

    parser.add_argument('--input', nargs='?', required=True,
                      help='Input graph file')

    parser.add_argument("-l", "--log", dest="log", default="INFO",
                      help="log verbosity level")

    parser.add_argument('--matfile-variable-name', default='network',
                      help='variable name of adjacency matrix inside a .mat file.')

    parser.add_argument('--max-memory-data-size', default=1000000000, type=int,
                      help='Size to start dumping walks to disk, instead of keeping them in memory.')

    parser.add_argument('--number-walks', default=10, type=int,
                      help='Number of random walks to start at each node')

    parser.add_argument('--output', required=True,
                      help='Output representation file')

    parser.add_argument('--representation-size', default=2, type=int,
                      help='Number of latent dimensions to learn for each node.')

    parser.add_argument('--seed', default=0, type=int,
                      help='Seed for random walk generator.')

    parser.add_argument('--undirected', default=True, type=bool,
                      help='Treat graph as undirected.')

    parser.add_argument('--vertex-freq-degree', default=False, action='store_true',
                      help='Use vertex degree to estimate the frequency of nodes '
                           'in the random walks. This option is faster than '
                           'calculating the vocabulary.')

    parser.add_argument('--walk-length', default=20, type=int,
                      help='Length of the random walk started at each node')

    parser.add_argument('--window-size', default=3, type=int,
                      help='Window size of skipgram model.')

    parser.add_argument('--workers', default=1, type=int,
                      help='Number of parallel processes.')

    args = parser.parse_args()
    numeric_level = getattr(logging, args.log.upper(), None)
    logging.basicConfig(format=LOGFORMAT)
    logger.setLevel(numeric_level)

    if args.debug:
        sys.excepthook = debug

    process(args)

if __name__ == "__main__":
    sys.exit(main())
