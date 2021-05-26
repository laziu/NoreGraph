import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(123)  # nopep8

import numpy as np
np.random.seed(123)  # nopep8
import time

from pytorch_U2GNN_UnSup import *
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from scipy.sparse import coo_matrix
from util import *
from sklearn.linear_model import LogisticRegression
import statistics
import json
import pickle
from argparse import Namespace

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)

parser = ArgumentParser("U2GNN_KAGGLE_TEST", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')

parser.add_argument("--run_folder", default="../", help="")
parser.add_argument("--model_name", default='PTC', help="")
args = parser.parse_args()

out_dir = os.path.abspath(os.path.join(args.run_folder, "../runs_pytorch_U2GNN_UnSup", args.model_name))
pth_path = os.path.abspath(os.path.join(out_dir, 'model.pth'))

with open(os.path.abspath(os.path.join(out_dir, 'args.json')), 'r') as f:
    dump = json.load(f)
    _args = vars(args)
    _args.update(dump)
    _args.update(vars(args))
    args = Namespace(**_args)

print(args)

# Load data
print("Loading data...")

try:
    with open(f'../../dataset_KAGGLE_True.pkl', 'rb') as f:
        graphs, num_classes, label_map, graph_name_map = pickle.load(f)
except IOError:
    graphs, num_classes, label_map, _, graph_name_map = load_data('KAGGLE', True)
    with open(f'../../dataset_KAGGLE_True.pkl', 'wb') as f:
        pickle.dump((graphs, num_classes, label_map, graph_name_map), f)

graph_labels = np.array([graph.label for graph in graphs])
feature_dim_size = graphs[0].node_features.shape[1]
print(feature_dim_size)


def get_Adj_matrix(batch_graph):
    edge_mat_list = []
    start_idx = [0]
    for i, graph in enumerate(batch_graph):
        start_idx.append(start_idx[i] + len(graph.g))
        edge_mat_list.append(graph.edge_mat + start_idx[i])

    Adj_block_idx = np.concatenate(edge_mat_list, 1)
    # Adj_block_elem = np.ones(Adj_block_idx.shape[1])

    Adj_block_idx_row = Adj_block_idx[0, :]
    Adj_block_idx_cl = Adj_block_idx[1, :]

    return Adj_block_idx_row, Adj_block_idx_cl


def get_graphpool(batch_graph):
    start_idx = [0]
    # compute the padded neighbor list
    for i, graph in enumerate(batch_graph):
        start_idx.append(start_idx[i] + len(graph.g))

    idx = []
    elem = []
    for i, graph in enumerate(batch_graph):
        elem.extend([1] * len(graph.g))
        idx.extend([[i, j] for j in range(start_idx[i], start_idx[i + 1], 1)])

    elem = torch.FloatTensor(elem)
    idx = torch.LongTensor(idx).transpose(0, 1)
    graph_pool = torch.sparse.FloatTensor(idx, elem, torch.Size([len(batch_graph), start_idx[-1]]))

    return graph_pool.to(device)


#
graph_pool = get_graphpool(graphs)
graph_indices = graph_pool._indices()[0]
vocab_size = graph_pool.size()[1]


def get_idx_nodes(selected_graph_idx):
    idx_nodes = [torch.where(graph_indices == i)[0] for i in selected_graph_idx]
    idx_nodes = torch.cat(idx_nodes)
    return idx_nodes.to(device)


def get_batch_data(selected_idx):
    batch_graph = [graphs[idx] for idx in selected_idx]

    X_concat = np.concatenate([graph.node_features for graph in batch_graph], 0)
    if "REDDIT" in args.dataset:
        X_concat = np.tile(X_concat, feature_dim_size)  # [1,1,1,1]
        X_concat = X_concat * 0.01
    X_concat = torch.from_numpy(X_concat).to(device)

    Adj_block_idx_row, Adj_block_idx_cl = get_Adj_matrix(batch_graph)
    dict_Adj_block = {}
    for i in range(len(Adj_block_idx_row)):
        if Adj_block_idx_row[i] not in dict_Adj_block:
            dict_Adj_block[Adj_block_idx_row[i]] = []
        dict_Adj_block[Adj_block_idx_row[i]].append(Adj_block_idx_cl[i])

    input_neighbors = []
    for input_node in range(X_concat.shape[0]):
        if input_node in dict_Adj_block:
            input_neighbors.append(
                [input_node] + list(np.random.choice(dict_Adj_block[input_node], args.num_neighbors, replace=True)))
        else:
            input_neighbors.append([input_node for _ in range(args.num_neighbors + 1)])
    input_x = np.array(input_neighbors)
    input_x = torch.from_numpy(input_x).long().to(device)

    input_y = get_idx_nodes(selected_idx)

    return X_concat, input_x, input_y


class Batch_Loader(object):
    def __call__(self):
        selected_idx = np.random.permutation(len(graphs))[:args.batch_size]
        X_concat, input_x, input_y = get_batch_data(selected_idx)
        return X_concat, input_x, input_y


batch_nodes = Batch_Loader()

print("Loading data... finished!")

model = TransformerU2GNN(feature_dim_size=feature_dim_size, ff_hidden_size=args.ff_hidden_size,
                         dropout=args.dropout, num_self_att_layers=args.num_timesteps,
                         vocab_size=vocab_size, sampled_num=args.sampled_num,
                         num_U2GNN_layers=args.num_hidden_layers, device=device).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
num_batches_per_epoch = int((len(graphs) - 1) / args.batch_size) + 1
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=num_batches_per_epoch, gamma=0.1)


def inspect():
    model.eval()
    with torch.no_grad():
        node_embeddings = model.ss.weight
        graph_embeddings = torch.spmm(graph_pool, node_embeddings).data.cpu().numpy()

        train_idx = [idx for idx, graph in enumerate(graphs) if graph.label is not None]
        train_graph_embeddings = graph_embeddings[train_idx]
        train_labels = graph_labels[train_idx].astype(int)

        cls = LogisticRegression(solver="lbfgs", tol=0.001, max_iter=1000)
        cls.fit(train_graph_embeddings, train_labels)

        out_path = os.path.abspath(os.path.join(out_dir, 'test_sample.csv'))
        with open('../../data/test.txt', 'r') as fi, open(out_path, 'w') as fo:
            fo.write('Id,Category\n')
            for line in fi:
                test_idx = [int(w) for w in re.findall(r'\d+', line)][0]
                test_internal_idx = [graph_name_map[test_idx]]
                test_graph_embedding = graph_embeddings[test_internal_idx]
                test_label_estimated = cls.predict(test_graph_embedding).item()
                if test_label_estimated in label_map:
                    test_label_estimated = label_map[test_label_estimated]
                fo.write(f'{test_idx},{test_label_estimated}\n')


"""main process"""
print("Reading {}\n".format(out_dir))
pth_path = os.path.abspath(os.path.join(out_dir, 'model.pth'))

model.load_state_dict(torch.load(pth_path))
print('Generating outputs')
inspect()
print('Finished')
