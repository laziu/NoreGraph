#! /usr/bin/env python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import time

from pytorch_U2GNN_Sup import *
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from scipy.sparse import coo_matrix
from util import *
import json
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
# ==================================================

parser = ArgumentParser("U2GNN", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')

parser.add_argument("--run_folder", default="../", help="")
parser.add_argument("--dataset", default="PTC", help="Name of the dataset.")
parser.add_argument("--learning_rate", default=0.0005, type=float, help="Learning rate")
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
graphs, num_classes, label_map, _, graph_name_map = load_cached_data(args.dataset)

weird = [graph.centrality_weirdness for graph in graphs]
size = [len(graph.g) for graph in graphs]
weird = torch.Tensor(weird).reshape(-1, 1) # batch size
size  = torch.Tensor(size ).reshape(-1, 1) # batch size
additional_info = torch.cat((weird, size), dim = 1)

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

    Adj_block_idx_row = Adj_block_idx[0,:]
    Adj_block_idx_cl = Adj_block_idx[1,:]

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

def get_batch_data(selected_idx):
    batch_graph = [graphs[idx] for idx in selected_idx]
    c_concat = np.concatenate([graph.centrality for graph in batch_graph], 0)
    c_concat = torch.from_numpy(c_concat).to(torch.float32)
    X_concat = np.concatenate([graph.node_features for graph in batch_graph], 0)
    if "REDDIT" in args.dataset:
        X_concat = np.tile(X_concat, feature_dim_size) #[1,1,1,1]
        X_concat = X_concat * 0.01
    X_concat = torch.from_numpy(X_concat)
    # graph-level sum pooling
    graph_pool = get_graphpool(batch_graph)

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
            input_neighbors.append([input_node for _ in range(args.num_neighbors+1)])
    input_x = np.array(input_neighbors)
    input_x = torch.from_numpy(input_x).long()
    #
    graph_labels = np.array([graph.label or 0 for graph in batch_graph]).astype(int)
    graph_labels = torch.from_numpy(graph_labels)

    return selected_idx, input_x, X_concat, graph_labels, c_concat

class IndexDataset(Dataset):
    def __init__(self, idx):
        super().__init__()
        self.idx = idx
    def __len__(self): return len(self.idx)
    def __getitem__(self, i): return self.idx[i]

trainidxset = IndexDataset([i for i, graph in enumerate(graphs) if graph.label is not None])
trainloader = DataLoader(trainidxset, batch_size=args.batch_size, shuffle=True,  collate_fn=get_batch_data, pin_memory=True)

print("Loading data... finished!")

model = TransformerU2GNN(feature_dim_size=feature_dim_size, ff_hidden_size=args.ff_hidden_size,
                        num_classes=num_classes, dropout=args.dropout,
                        num_self_att_layers=args.num_timesteps,
                        num_U2GNN_layers=args.num_hidden_layers).to(device)

def cross_entropy(pred, soft_targets): # use nn.CrossEntropyLoss if not using soft labels in Line 159
    logsoftmax = nn.LogSoftmax(dim=1)
    return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))

# criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
num_batches_per_epoch = int((len(graphs) - 1) / args.batch_size) + 1
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=num_batches_per_epoch, gamma=0.1)

def train():
    model.train() # Turn on the train mode
    total_loss = 0.
    total_correct = 0
    for selected_idx, input_x, X_concat, graph_labels, c_concat in tqdm(trainloader, desc='train'):
        input_x = input_x.to(device)
        X_concat = X_concat.to(device)
        graph_labels = graph_labels.to(device)
        c_concat = c_concat.to(device)
        graph_pool = get_graphpool([graphs[idx] for idx in selected_idx]).to(device)
        smoothed_labels = label_smoothing(graph_labels, num_classes)
        optimizer.zero_grad()
        prediction_scores = model(input_x, graph_pool, X_concat, c_concat)
        # loss = criterion(prediction_scores, graph_labels)
        loss = cross_entropy(prediction_scores, smoothed_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) # prevent the exploding gradient problem
        optimizer.step()
        total_loss += loss.item()

        predictions = prediction_scores.max(1, keepdim=True)[1]
        total_correct += predictions.eq(graph_labels.view_as(predictions)).sum().cpu().item()

    acc_test = total_correct / float(len(trainidxset))
    return total_loss, acc_test

def inspect():
    model.eval()
    with torch.no_grad():
        prediction_output = []
        csv_idx = []
        idx = []
        with open(project_root/'data/test.txt', 'r') as fi:
            for line in fi:
                test_idx = [int(w) for w in re.findall(r'\d+', line)][0]
                test_internal_idx = graph_name_map[test_idx]
                csv_idx.append(test_idx)
                idx.append(test_internal_idx)
        idx = np.array(idx)
        for i in range(0, len(idx), args.batch_size):
            selected_idx = idx[i:i + args.batch_size]
            if len(selected_idx) == 0: continue
            selected_idx, input_x, X_concat, _, c_concat = get_batch_data(selected_idx)
            graph_pool = get_graphpool([graphs[idx] for idx in selected_idx])
            prediction_scores = model(input_x.to(device), graph_pool.to(device), X_concat.to(device), c_concat.to(device)).detach()
            prediction_output.append(prediction_scores)
        prediction_output = torch.cat(prediction_output, 0)
        predictions = prediction_output.max(1, keepdim=True)[1]
        labels_est = [l.item() for l in predictions.cpu().squeeze()]
        labels_est = [(label_map[l] if l in label_map else l) for l in labels_est]
        with open(out_dir/'test_sample.csv', 'w') as fo:
            fo.write('Id,Category\n')
            for t_idx, label in zip(csv_idx, labels_est):
                fo.write(f'{t_idx},{label}\n')

"""main process"""
out_dir: Path = project_root/"runs_pytorch_U2GNN_Sup"/args.model_name
out_dir.mkdir(exist_ok=True, parents=True)
print("Writing to {}\n".format(out_dir))
# Checkpoint directory
checkpoint_dir = out_dir/"checkpoints"
checkpoint_dir.mkdir(exist_ok=True, parents=True)
write_acc = open(checkpoint_dir/'model_acc.txt', 'w')
pth_path = out_dir/'model.pth'

if not args.only_test:
    cost_loss = []
    for epoch in range(1, args.num_epochs + 1):
        train_graphs, test_graphs = separate_data(graphs, args.fold_idx, None)

        epoch_start_time = time.time()
        train_loss, acc_test = train()
        cost_loss.append(train_loss)
        print('| epoch {:3d} | time: {:5.2f}s | loss {:5.2f} | test acc {:5.2f} | '.format(
                    epoch, (time.time() - epoch_start_time), train_loss, acc_test*100))

        if epoch > 5 and cost_loss[-1] > np.mean(cost_loss[-6:-1]):
            scheduler.step()

        write_acc.write('epoch ' + str(epoch) + ' fold ' + str(args.fold_idx) + ' acc ' + str(acc_test*100) + '%\n')
        torch.save(model.state_dict(), pth_path)
        if not args.only_train:
            inspect()
else:
    model.load_state_dict(torch.load(pth_path))
    inspect()

write_acc.close()
