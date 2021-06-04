import networkx as nx
import numpy as np
import re
import random
import scipy.sparse as sp
from sklearn.model_selection import StratifiedKFold
import pickle
from pathlib import Path
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from multiprocessing import cpu_count

project_root = Path(__file__).parent.absolute()
current_root = Path(__file__).parent.absolute()

"""Adapted from https://github.com/weihua916/powerful-gnns/blob/master/util.py"""


class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        '''
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = 0
        self.edge_mat = 0
        self.max_neighbor = 0
        self.name = None
        self.centrality = []
        self.centrality_weirdness = 0


def _post_processing(g: S2VGraph):
    g.neighbors = [[] for i in range(len(g.g))]
    for i, j in g.g.edges():
        g.neighbors[i].append(j)
        g.neighbors[j].append(i)
    degree_list = []
    for i in range(len(g.g)):
        g.neighbors[i] = g.neighbors[i]
        degree_list.append(len(g.neighbors[i]))
    g.max_neighbor = max(degree_list)

    edges = [list(pair) for pair in g.g.edges()]
    edges.extend([[i, j] for j, i in edges])

    #deg_list = list(dict(g.g.degree(range(len(g.g)))).values())

    g.edge_mat = np.transpose(np.array(edges, dtype=np.int32), (1, 0))

    b_cent = list(nx.betweenness_centrality(g.g).values())
    c_cent = list(nx.closeness_centrality(g.g).values())
    d_cent = list(nx.degree_centrality(g.g).values())
    g.centrality = np.stack([b_cent, c_cent, d_cent], axis=1)

    weirdness = (max(map(float, b_cent)) < 0.00001) or \
                (min(map(float, c_cent)) > 0.99) or \
                (min(map(float, d_cent)) > 0.99)
    g.centrality_weirdness = float(weirdness)
    return g


def load_data(dataset):
    '''
        dataset: name of dataset
        test_proportion: ratio of test train split
    '''
    degree_as_tag = False
    if str(dataset).upper() in ['COLLAB', 'IMDBBINARY', 'IMDBMULTI', 'KAGGLE']:
        degree_as_tag = True

    print('loading data')
    g_list = []
    label_dict = {}
    feat_dict = {}

    if dataset.upper() == 'KAGGLE':
        node_features = None
        node_feature_flag = False

        g_dict = {}  # graph_index -> nx.Graph
        l_dict = {}  # graph_index -> label
        n2g_dict = {}  # node_index -> graph_index
        ngi_dict = {}  # node_index -> index in graph

        with open(project_root/'data/train.txt', 'r') as f:
            for line in tqdm(f, desc='reading data/train.txt'):
                row = re.findall(r'\d+', line)
                gi, l = [int(w) for w in row]
                if not l in label_dict:
                    mapped = len(label_dict)
                    label_dict[l] = mapped
                g_dict[gi] = nx.Graph()
                l_dict[gi] = l

        with open(project_root/'data/test.txt', 'r') as f:
            for line in tqdm(f, desc='reading data/test.txt'):
                row = re.findall(r'\d+', line)
                gi = int(row[0])
                g_dict[gi] = nx.Graph()
                l_dict[gi] = None

        with open(project_root/'data/graph_ind.txt', 'r') as f:
            for line in tqdm(f, desc='reading data/graph_ind.txt'):
                row = re.findall(r'\d+', line)
                ni, gi = [int(w) for w in row]
                if gi in g_dict:
                    g: nx.Graph = g_dict[gi]
                    ngi = len(g.nodes)
                    g.add_node(ngi)
                    n2g_dict[ni] = gi
                    ngi_dict[ni] = ngi

        with open(project_root/'data/graph.txt', 'r') as f:
            for line in tqdm(f, desc='reading data/graph.txt'):
                row = re.findall(r'\d+', line)
                ni, nj = [int(w) for w in row]
                if ni in n2g_dict:
                    assert nj in n2g_dict
                    gi, gj = n2g_dict[ni], n2g_dict[nj]
                    ngi, ngj = ngi_dict[ni], ngi_dict[nj]
                    assert gi in g_dict
                    assert gi == gj
                    g: nx.Graph = g_dict[gi]
                    g.add_edge(ngi, ngj)

        for gi in g_dict:
            g = g_dict[gi]
            l = l_dict[gi]
            graph = S2VGraph(g, l, [0 for _ in g.nodes])
            graph.name = gi
            g_list.append(graph)

    else:
        with open(project_root/f'Graph-Transformer/dataset/{dataset}/{dataset}.txt', 'r') as f:
            n_g = int(f.readline().strip())
            for i in tqdm(range(n_g), desc=f'reading {dataset}.txt'):
                row = f.readline().strip().split()
                n, l = [int(w) for w in row]
                if not l in label_dict:
                    mapped = len(label_dict)
                    label_dict[l] = mapped
                g = nx.Graph()
                node_tags = []
                node_features = []
                n_edges = 0
                for j in range(n):
                    g.add_node(j)
                    row = f.readline().strip().split()
                    tmp = int(row[1]) + 2
                    if tmp == len(row):
                        # no node attributes
                        row = [int(w) for w in row]
                        attr = None
                    else:
                        row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
                    if not row[0] in feat_dict:
                        mapped = len(feat_dict)
                        feat_dict[row[0]] = mapped
                    node_tags.append(feat_dict[row[0]])

                    if tmp > len(row):
                        node_features.append(attr)

                    n_edges += row[1]
                    for k in range(2, len(row)):
                        g.add_edge(j, row[k])

                if node_features != []:
                    node_features = np.stack(node_features)
                    node_feature_flag = True
                else:
                    node_features = None
                    node_feature_flag = False

                assert len(g) == n

                g_list.append(S2VGraph(g, l, node_tags))

    # add labels and edge_mat
    for g in g_list:
        g.label = label_dict[g.label] if g.label in label_dict else None

    g_list = process_map(_post_processing, g_list, desc="postprocessing", chunksize=4, max_workers=(cpu_count() - 2))

    if degree_as_tag:
        for g in g_list:
            g.node_tags = list(dict(g.g.degree).values())

    # Extracting unique tag labels
    tagset = set([])
    for g in g_list:
        tagset = tagset.union(set(g.node_tags))

    tagset = list(tagset)
    tag2index = {tagset[i]: i for i in range(len(tagset))}

    for g in g_list:
        g.node_features = np.zeros((len(g.node_tags), len(tagset)), dtype=np.float32)
        g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1

    print('# classes: %d' % len(label_dict))
    print('# maximum node tag: %d' % len(tagset))

    print("# data: %d" % len(g_list))

    label_map = {idx: label for label, idx in label_dict.items()}
    feat_map = {idx: feat for feat, idx in feat_dict.items()}
    graph_map = {graph.name: idx for idx, graph in enumerate(g_list) if graph.name is not None}

    return g_list, len(label_dict), label_map, feat_map, graph_map


def load_cached_data(dataset):
    try:
        with open(project_root/f"dataset_{dataset}.pkl", "rb") as f:
            chunk = pickle.load(f)
    except IOError:
        chunk = load_data(dataset)
        with open(project_root/f"dataset_{dataset}.pkl", "wb") as f:
            pickle.dump(chunk, f)
    return chunk


def separate_data(graph_list, fold_idx, seed=0):
    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

    indices, labels = np.transpose([[int(i), int(graph.label)]
                                    for i, graph in enumerate(graph_list)
                                    if graph.label is not None])
    idx_list = []
    idx_list = list(skf.split(np.zeros(len(labels)), labels))
    train_idx, test_idx = idx_list[fold_idx]

    train_graph_list = [graph_list[i] for i in indices[train_idx]]
    test_graph_list = [graph_list[i] for i in indices[test_idx]]

    return train_graph_list, test_graph_list


def separate_data_idx(graph_list, fold_idx, seed=0):
    """ Get indexes of train and test sets """
    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

    indices, labels = np.transpose([[int(i), int(graph.label)]
                                    for i, graph in enumerate(graph_list)
                                    if graph.label is not None])

    idx_list = list(skf.split(np.zeros(len(labels)), labels))
    train_idx, test_idx = idx_list[fold_idx]

    return indices[train_idx], indices[test_idx]


def sparse_to_tuple(sparse_mx):
    """ Convert sparse matrix to tuple representation. """
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx