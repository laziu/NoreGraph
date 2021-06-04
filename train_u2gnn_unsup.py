#! /usr/bin/env python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import Dataset, DataLoader

import numpy as np
import time

from u2gnn.model_unsup import *
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from scipy.sparse import coo_matrix
from util import *
from sklearn.linear_model import LogisticRegression
import statistics
import json
import pickle
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
# ==================================================

parser = ArgumentParser("U2GNN", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')

parser.add_argument("--run_folder", default="../", help="")
parser.add_argument("--dataset", default="PTC", help="Name of the dataset.")
parser.add_argument("--learning_rate", default=0.005, type=float, help="Learning rate")
parser.add_argument("--batch_size", default=4, type=int, help="Batch Size")
parser.add_argument("--num_epochs", default=50, type=int, help="Number of training epochs")
parser.add_argument("--model_name", default='PTC', help="")
parser.add_argument('--sampled_num', default=512, type=int, help='')
parser.add_argument("--dropout", default=0.5, type=float, help="")
parser.add_argument("--num_hidden_layers", default=1, type=int, help="")
parser.add_argument("--num_timesteps", default=1, type=int, help="Timestep T ~ Number of self-attention layers within each U2GNN layer")
parser.add_argument("--ff_hidden_size", default=1024, type=int, help="The hidden size for the feedforward layer")
parser.add_argument("--num_neighbors", default=4, type=int, help="")
parser.add_argument('--fold_idx', type=int, default=1, help='The fold index. 0-9.')
parser.add_argument('--only_train', type=bool, default=False)
parser.add_argument('--only_test', type=bool, default=False)
args = parser.parse_args()

print(args)

# Load data
print("Loading data...")
graphs, _, label_map, _, graph_name_map = load_cached_data(args.dataset)

weird = [graph.centrality_weirdness for graph in graphs]
size = [len(graph.g) for graph in graphs]
weird = torch.Tensor(weird).reshape(-1, 1)  # batch size
size = torch.Tensor(size).reshape(-1, 1)  # batch size
additional_info = torch.cat((weird, size), dim=1)

graph_labels = np.array([graph.label for graph in graphs])
feature_dim_size = graphs[0].node_features.shape[1] + graphs[0].centrality.shape[1]
print(feature_dim_size)
if "REDDIT" in args.dataset:
    feature_dim_size = 4


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

    return graph_pool


#
graph_pool = get_graphpool(graphs)
graph_indices = graph_pool._indices()[0]
vocab_size = graph_pool.size()[1]


def get_idx_nodes(selected_graph_idx):
    idx_nodes = [torch.where(graph_indices == i)[0] for i in selected_graph_idx]
    idx_nodes = torch.cat(idx_nodes)
    return idx_nodes


def get_batch_data(selected_idx):
    batch_graph = [graphs[idx] for idx in selected_idx]
    c_concat = np.concatenate([graph.centrality for graph in batch_graph], 0)
    c_concat = torch.from_numpy(c_concat).to(torch.float32)
    # print('c_concat',c_concat.shape)
    X_concat = np.concatenate([graph.node_features for graph in batch_graph], 0)
    if "REDDIT" in args.dataset:
        X_concat = np.tile(X_concat, feature_dim_size)  # [1,1,1,1]
        X_concat = X_concat * 0.01
    X_concat = torch.from_numpy(X_concat)

    Adj_block_idx_row, Adj_block_idx_cl = get_Adj_matrix(batch_graph)
    dict_Adj_block = {}
    for i in range(len(Adj_block_idx_row)):
        if Adj_block_idx_row[i] not in dict_Adj_block:
            dict_Adj_block[Adj_block_idx_row[i]] = []
        dict_Adj_block[Adj_block_idx_row[i]].append(Adj_block_idx_cl[i])

    input_neighbors = []
    for input_node in range(X_concat.shape[0]):
        if input_node in dict_Adj_block:
            input_neighbors.append([input_node] + list(np.random.choice(dict_Adj_block[input_node], args.num_neighbors, replace=True)))
        else:
            input_neighbors.append([input_node for _ in range(args.num_neighbors + 1)])
    input_x = np.array(input_neighbors)
    input_x = torch.from_numpy(input_x).long()

    input_y = get_idx_nodes(selected_idx)

    return X_concat, input_x, input_y, c_concat


class IndexDataset(Dataset):
    def __len__(self): return len(graphs)
    def __getitem__(self, i): return i


dataloader = DataLoader(IndexDataset(), batch_size=args.batch_size, shuffle=True, collate_fn=get_batch_data, pin_memory=True)

print("Loading data... finished!")

model = UnsupU2GNN(feature_dim_size=feature_dim_size, ff_hidden_size=args.ff_hidden_size,
                   dropout=args.dropout, num_self_att_layers=args.num_timesteps,
                   vocab_size=vocab_size, sampled_num=args.sampled_num,
                   num_U2GNN_layers=args.num_hidden_layers, device=device).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
num_batches_per_epoch = int((len(graphs) - 1) / args.batch_size) + 1
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=num_batches_per_epoch, gamma=0.1)


def train():
    model.train()  # Turn on the train mode
    total_loss = 0.
    for X_concat, input_x, input_y, c_concat in tqdm(dataloader, desc='train'):
        optimizer.zero_grad()
        logits = model(X_concat.to(device), input_x.to(device), input_y.to(device), c_concat.to(device))
        loss = torch.sum(logits)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()

    return total_loss


def evaluate():
    model.eval()  # Turn on the evaluation mode
    with torch.no_grad():
        # evaluating
        node_embeddings = model.ss.weight
        graph_embeddings = torch.spmm(graph_pool, node_embeddings.cpu()).data.numpy()
        graph_embeddings = np.concatenate((graph_embeddings, additional_info.numpy()), 1)
        print(graph_embeddings.shape)

        acc_10folds = []
        rand_seed = np.random.randint(0xFFFF)
        for fold_idx in range(10):
            train_idx, test_idx = separate_data_idx(graphs, fold_idx, rand_seed)
            train_graph_embeddings = graph_embeddings[train_idx]
            test_graph_embeddings = graph_embeddings[test_idx]
            train_labels = graph_labels[train_idx].astype(int)
            test_labels = graph_labels[test_idx].astype(int)

            cls = LogisticRegression(solver="liblinear", tol=0.001)
            cls.fit(train_graph_embeddings, train_labels)
            ACC = cls.score(test_graph_embeddings, test_labels)
            acc_10folds.append(ACC)
        print(f'evaluate: {" ".join([f"{acc*100:.3f}" for acc in acc_10folds])}')

        mean_10folds = statistics.mean(acc_10folds)
        std_10folds = statistics.stdev(acc_10folds)
        # print('epoch ', epoch, ' mean: ', str(mean_10folds), ' std: ', str(std_10folds))

    return mean_10folds, std_10folds

def inspect():
    model.eval()
    with torch.no_grad():
        node_embeddings = model.ss.weight
        graph_embeddings = torch.spmm(graph_pool, node_embeddings.cpu()).data.numpy()
        graph_embeddings = np.concatenate((graph_embeddings, additional_info.numpy()), 1)

        train_idx = [idx for idx, graph in enumerate(graphs) if graph.label is not None]
        train_graph_embeddings = graph_embeddings[train_idx]
        train_labels = graph_labels[train_idx].astype(int)

        cls = LogisticRegression(solver="liblinear", tol=0.001)
        cls.fit(train_graph_embeddings, train_labels)

        with open(project_root/'data/test.txt', 'r') as fi, open(out_dir/'test_sample.csv', 'w') as fo:
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
out_dir: Path = project_root/"runs_pytorch_U2GNN_UnSup"/args.model_name
out_dir.mkdir(exist_ok=True, parents=True)
print("Writing to {}\n".format(out_dir))
# Checkpoint directory
checkpoint_dir = out_dir/"checkpoints"
checkpoint_dir.mkdir(exist_ok=True, parents=True)
write_acc = open(checkpoint_dir/'model_acc.txt', 'w')
pth_path = out_dir/'model.pth'

with open(out_dir/'args.json', 'w') as f:
    json.dump(vars(args), f, indent=4)

if not args.only_test:
    cost_loss = []
    for epoch in range(1, args.num_epochs + 1):
        epoch_start_time = time.time()
        train_loss = train()
        cost_loss.append(train_loss)
        mean_10folds, std_10folds = evaluate()
        print('| epoch {:3d} | time: {:5.2f}s | loss {:5.2f} | mean {:5.2f} | std {:5.2f} | '.format(
            epoch, (time.time() - epoch_start_time), train_loss, mean_10folds*100, std_10folds*100))

        if epoch > 5 and cost_loss[-1] > np.mean(cost_loss[-6:-1]):
            scheduler.step()

        write_acc.write('epoch ' + str(epoch) + ' mean: ' + str(mean_10folds*100) + ' std: ' + str(std_10folds*100) + '\n')
        torch.save(model.state_dict(), pth_path)
        if not args.only_train:
            inspect()
else:
    model.load_state_dict(torch.load(pth_path))
    inspect()

write_acc.close()
