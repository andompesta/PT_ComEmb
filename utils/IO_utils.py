__author__ = 'ando'

import pickle
import numpy as np
from os.path import join as path_join, dirname
from os import makedirs

def save_ground_true(path, community_color):
    with open(path, 'w') as txt_file:
        for node, com in enumerate(community_color):
            txt_file.write('%d\t%d\n' % ((node+1), com))

def load_ground_true(path='./data', file_name=None):
    temp = []
    with open(path_join(path, file_name + '.labels'), 'r') as file:
        for line_no, line in enumerate(file):
            tokens = line.strip().split('\t')
            temp.append(int(tokens[1]))
    ground_true = np.array(temp, dtype=np.uint8)
    k = max(ground_true)
    return ground_true, k

def save_embedding(embeddings, file_name, path='./data'):
    with open(path_join(path, file_name + '.txt'), 'w') as file:
        for node_id, embed in enumerate(embeddings):
            file.write(str(node_id+1) + '\t' + " ".join([str(val) for val in embed]) + '\n')

def load_embedding(file_name, path='data', ext=".txt"):
    ret = []
    with open(path_join(path, file_name + ext), 'r') as file:
        for line in file:
            tokens = line.strip().split('\t')
            node_values = [float(val) for val in tokens[1].strip().split(' ')]
            ret.append(node_values)
    ret = np.array(ret, dtype=np.float32)
    return ret

def save_membership(membership_matrix, file_name, path='data'):
    with open(path + '/' + file_name + '.txt', 'w') as file:
        for node_id, value in enumerate(membership_matrix):
            file.write('%d\t%d\n' % (node_id, value))

def load_membership(file_name, path='data'):
    membership = []
    with open(path + '/' + file_name + '.txt', 'r') as file:
        for line in file:
            tokens = line.strip().split('\t')
            membership.append(int(tokens[1]))
    return membership



def save(data, file_name, path='data'):
    '''
    Dump datastructure with pickle
    :param data: data to dump
    :param file_name: file name
    :param path: dire where to save the file
    :return:
    '''
    full_path = path_join(path, file_name + '.bin')
    makedirs(dirname(full_path), exist_ok=True)
    with open(full_path, 'wb') as file:
        pickle.dump(data, file)


def load(file_name, path='data'):
    '''
    Load datastructure with pickle
    :param file_name: file name
    :param path: dire where to save the file
    :return:
    '''
    full_path = path_join(path, file_name)
    with open(full_path, 'rb') as file:
        data = pickle.load(file)
    return data