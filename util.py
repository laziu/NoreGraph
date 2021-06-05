import os
import errno
from pathlib import Path
import re
import pickle

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from multiprocessing import cpu_count

import numpy as np
import networkx as nx
import scipy.sparse as sp
from sklearn.model_selection import StratifiedKFold

from typing import Optional

project_root = Path(__file__).parent.absolute()
dataset_root = project_root/f"data"


class S2VGraph:
    def __init__(self, g: nx.Graph, index: int, label: Optional[int]):
        """
        Args:
            g: a networkx graph
            index: graph index in dataset
            label: an integer graph label in internal representation
            real_label: real graph label in dataset
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            node_centrality: centrality of nodes
            graph_features: features for whole graph
            neighbors: list of neighbors (without self-loop)
            max_neighbor: maximum number of neighbors for all nodes
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
        """
        self.g = g
        self.index = index
        self.label = label

        self.node_tags = np.array(list(dict(g.degree).values()))

        self.node_features = np.array([])  # not set

        cent_fn = [nx.betweenness_centrality, nx.closeness_centrality, nx.degree_centrality]
        self.node_centrality = np.stack([list(f(g).values()) for f in cent_fn], axis=1)

        num_nodes = len(g)
        centrality_weirdness = int((max(map(float, self.node_centrality[0])) < 0.00001) or
                                   (min(map(float, self.node_centrality[1])) > 0.99) or
                                   (min(map(float, self.node_centrality[2])) > 0.99))
        self.graph_features = np.array([num_nodes, centrality_weirdness])

        self.neighbors = [[] for i in range(num_nodes)]
        for i, j in g.edges():
            self.neighbors[i].append(j)
            self.neighbors[j].append(i)

        self.max_neighbor = max([len(self.neighbors[i]) for i in range(num_nodes)])

        edges = np.array(g.edges(), dtype=np.int32)
        edges = np.concatenate((edges, edges[:, [1, 0]]))
        self.edge_mat = np.transpose(edges, (1, 0))


def datafile(name, mode="r"):
    fpath = dataset_root/name
    if "r" in mode and not fpath.is_file():
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), f"data/{name}")
    return open(fpath, mode)


def parse_text():
    li_map: dict[int, int] = {}       # real_label -> label
    g_dict: dict[int, nx.Graph] = {}  # graph_index -> nx.Graph
    l_dict: dict[int, int] = {}       # graph_index -> real_label
    node_g: dict[int, int] = {}       # node_index -> graph_index
    node_i: dict[int, int] = {}       # node_index -> node index in graph

    with datafile("train.txt") as f:
        for line in tqdm(f, desc="Reading data/train.txt"):
            gi, l = [int(w) for w in re.findall(r"\d+", line)]
            g_dict[gi] = nx.Graph()
            l_dict[gi] = l
            if not l in li_map:
                l_int = len(li_map)
                li_map[l] = l_int

    with datafile("test.txt") as f:
        for line in tqdm(f, desc="Reading data/test.txt"):
            gi = int(re.findall(r"\d+", line)[0])
            g_dict[gi] = nx.Graph()

    with datafile("graph_ind.txt") as f:
        for line in tqdm(f, desc="Reading data/graph_ind.txt"):
            ni, gi = [int(w) for w in re.findall(r"\d+", line)]
            assert gi in g_dict
            g: nx.Graph = g_dict[gi]
            ngi = len(g.nodes)
            g.add_node(ngi)
            node_g[ni] = gi
            node_i[ni] = ngi

    with datafile("graph.txt", "r") as f:
        for line in tqdm(f, desc="Reading data/graph.txt"):
            ni, nj = [int(w) for w in re.findall(r"\d+", line)]
            assert ni in node_g
            assert nj in node_g
            gi, gj = node_g[ni], node_g[nj]
            ngi, ngj = node_i[ni], node_i[nj]
            assert gi in g_dict
            assert gi == gj
            g: nx.Graph = g_dict[gi]
            g.add_edge(ngi, ngj)

    return g_dict, l_dict, li_map, node_g, node_i


def s2vgraph_construct(args): return S2VGraph(*args)


def load_data():
    g_dict, l_dict, li_map, _, _ = parse_text()

    g_args = [(g_dict[gi], gi, li_map.get(l_dict.get(gi))) for gi in g_dict]
    graphs: list[S2VGraph] = process_map(s2vgraph_construct, g_args, desc="Postprocessing", chunksize=4, max_workers=(cpu_count() - 2))

    tagset: list[int] = list(set().union(*[set(graph.node_tags) for graph in graphs]))
    tag2index = {tagset[i]: i for i in range(len(tagset))}

    for graph in graphs:
        graph.node_features = np.zeros((len(graph.node_tags), len(tagset)), dtype=np.float32)
        graph.node_features[range(len(graph.node_tags)), [tag2index[tag] for tag in graph.node_tags]] = 1

    label_map = {label: real_label for real_label, label in li_map.items()}
    graph_pos = {graph.index: pos for pos, graph in enumerate(graphs)}

    return graphs, label_map, graph_pos


def load_cached_data():
    chunk: tuple[list[S2VGraph], dict[int, int], dict[int, int]]
    try:
        with datafile(f"dataset.pkl", "rb") as f:
            chunk = pickle.load(f)
    except IOError:
        chunk = load_data()
        with datafile(f"dataset.pkl", "wb") as f:
            pickle.dump(chunk, f)

    graphs, label_map, _ = chunk
    maxtag = len(set().union(*[set(graph.node_tags) for graph in graphs]))
    print(f"# data: {len(graphs)} | # classes: {len(label_map)} | # max node tag: {maxtag}")

    return chunk


def train_idx(graphs: "list[S2VGraph]", labeled_only=True):
    return np.transpose([(i, graph.index)
                         for i, graph in enumerate(graphs)
                         if (not labeled_only) or (graph.label is not None)])


def kfold_train_idx(graphs: "list[S2VGraph]", n_splits=10, labeled_only=True, seed=None):
    """ Get indexes of train and test sets """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    indices, labels = np.transpose([[int(i), int(graph.label)]
                                    for i, graph in enumerate(graphs)
                                    if (not labeled_only) or (graph.label is not None)])
    idx_list = list(skf.split(np.zeros(len(labels)), labels))
    idx_list = [(indices[train_idx], indices[test_idx])
                for train_idx, test_idx in idx_list]
    return idx_list


def test_idx(graph_pos: "dict[int, int]"):
    with datafile("test.txt") as f:
        real_idx = [int(re.findall(r"\d+", line)[0]) for line in f]
        intr_idx = [graph_pos[i] for i in real_idx]
        return intr_idx, real_idx


def test_accuracy(index: "list[int]", label: "list[int]"):
    def collab_classifier(idx):
        return 1 if idx <= 2600 else 2 if idx <= 3375 else 3

    total = len(index)
    success = sum([l == collab_classifier(idx) for idx, l in zip(index, label)])
    return success/total


def batch(data: list, batch_size=1):
    return [data[i:i+batch_size] for i in range(0, len(data), batch_size)]


def shuffle(data: list):
    return np.random.permutation(data)


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
