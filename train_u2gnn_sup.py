#!/usr/bin/env python
from time import time
import json
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from tqdm import tqdm
import pandas as pd

import math
import numpy as np
from scipy.sparse import coo_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F

from u2gnn.model_sup import *
from util import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_name", default="KAGGLE", help="Output directory name")
    parser.add_argument("--learning_rate", default=0.0001, type=float, help="Learning rate")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size")
    parser.add_argument("--num_epochs", default=50, type=int, help="Number of training epochs")
    parser.add_argument("--hidden_size", default=128, type=int, help="The hidden size for the feedforward layer")
    parser.add_argument("--num_hidden_layers", default=1, type=int, help="Number of hidden layers in the encoder")
    parser.add_argument("--num_timesteps", default=1, type=int, help="Timestep T ~ Number of self-attention layers within each U2GNN layer")
    parser.add_argument("--num_neighbors", default=16, type=int, help="Number of neighbors for the input of the encoder")
    parser.add_argument("--dropout", default=0.5, type=float, help="Dropout rate")
    parser.add_argument("--load_epoch", default=0, type=int, help="Load previous state if set")
    parser.add_argument("--test_only", action='store_true', help="Print test result and exit")
    args = parser.parse_args()
    print(args)

    print("Loading data...")
    graphs: "list[S2VGraph]"
    label_map: "dict[int, int]"
    graph_pos: "dict[int, int]"
    graphs, label_map, graph_pos = load_cached_data()
    print("Loading data finished.")

    graph_features = torch.from_numpy(np.array([graph.graph_features for graph in graphs]))
    node_features_dim = graphs[0].node_features.shape[1] + graphs[0].node_centrality.shape[1]
    num_classes = len(label_map)
    print(f"node features dimension: {node_features_dim}")

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
        c_concat = np.concatenate([graph.node_centrality for graph in batch_graph], 0)
        X_concat = np.concatenate([graph.node_features for graph in batch_graph], 0)
        X_concat = np.concatenate((X_concat, c_concat), axis=1)
        X_concat = torch.from_numpy(X_concat).to(device, dtype=torch.float32)

        # graph-level sum pooling
        graph_pool = get_graphpool(batch_graph).to(device)

        edge_mat_list = []
        start_idx = [0]
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))
            edge_mat_list.append(graph.edge_mat + start_idx[i])

        Adj_block_idx = np.concatenate(edge_mat_list, 1)
        # Adj_block_elem = np.ones(Adj_block_idx.shape[1])
        Adj_block_idx_row = Adj_block_idx[0, :]
        Adj_block_idx_cl = Adj_block_idx[1, :]

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
        input_x = torch.from_numpy(input_x).to(device, dtype=torch.long)
        #
        graph_labels = np.array([graph.label or 0 for graph in batch_graph]).astype(int)
        graph_labels = torch.from_numpy(graph_labels).to(device)

        return X_concat, input_x, graph_pool, graph_labels

    model = SupU2GNN(feature_dim_size=node_features_dim, ff_hidden_size=args.hidden_size,
                     num_classes=num_classes, dropout=args.dropout,
                     num_self_att_layers=args.num_timesteps,
                     num_U2GNN_layers=args.num_hidden_layers).to(device)

    # criterion = nn.CrossEntropyLoss()
    def cross_entropy(pred, soft_targets):  # use nn.CrossEntropyLoss if not using soft labels in Line 159
        logsoftmax = nn.LogSoftmax(dim=1)
        return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    num_batches_per_epoch = int((len(graphs) - 1) / args.batch_size) + 1
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=num_batches_per_epoch, gamma=0.1)

    train_intr_idx, _ = train_idx(graphs, labeled_only=True)
    test_intr_idx, test_real_idx = test_idx(graph_pos)

    def train():
        model.train()  # Turn on the train mode
        train_loss = 0.
        correct = 0
        for selected_idx in tqdm(batch(shuffle(train_intr_idx), batch_size=args.batch_size), desc="train"):
            X_concat, input_x, graph_pool, graph_labels = get_batch_data(selected_idx)
            smoothed_labels = label_smoothing(graph_labels, num_classes)
            optimizer.zero_grad()
            prediction_scores = model(X_concat, input_x, graph_pool)
            # loss = criterion(prediction_scores, graph_labels)
            loss = cross_entropy(prediction_scores, smoothed_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # prevent the exploding gradient problem
            optimizer.step()
            train_loss += loss.item()

            predictions = prediction_scores.max(1, keepdim=True)[1]
            correct += predictions.eq(graph_labels.view_as(predictions)).sum().cpu().item()

        train_acc = correct / float(len(train_intr_idx))
        return train_acc, train_loss

    def test():
        model.eval()
        with torch.no_grad():
            prediction_output = []
            for selected_idx in batch(test_intr_idx, batch_size=args.batch_size):
                X_concat, input_x, graph_pool, _ = get_batch_data(selected_idx)
                prediction_scores = model(X_concat, input_x, graph_pool).detach()
                prediction_output.append(prediction_scores)
            prediction_output = torch.cat(prediction_output, 0)
            predictions = prediction_output.max(1, keepdim=True)[1]
            labels_est = [l.item() for l in predictions.cpu().squeeze()]
            labels_est = [(label_map[l] if l in label_map else l) for l in labels_est]

        test_acc = test_accuracy(test_real_idx, labels_est)
        return test_acc, labels_est

    """ main process """
    out_dir: Path = project_root/"runs/U2GNNsup"/args.model_name
    out_dir.mkdir(exist_ok=True, parents=True)
    print(f"Using {out_dir}")

    global_args_key = ["hidden_size", "num_hidden_layers", "num_timesteps", "dropout"]
    epoch_args_key = ["learning_rate", "batch_size", "num_neighbors"]
    checkpoints_key = ["time", "train_acc", "train_loss", "test_acc"]

    def filter_args(arg: dict, filter: list):
        return {k: v for k, v in arg.items() if k in filter}

    if (out_dir/"args.json").is_file():
        with open(out_dir/"args.json", "r") as f:
            cargs = filter_args(vars(args), global_args_key)
            sargs = filter_args(json.load(f), global_args_key)
            if cargs != sargs:
                print('args mismatch!')
                print('current args:', cargs)
                print('saved args:', sargs)
                raise ValueError(f"runs/U2GNNsup/{args.model_name}/args.json is not same with current args, delete it to dismiss.")
    else:
        with open(out_dir/"args.json", "w") as f:
            json.dump(filter_args(vars(args), global_args_key), f, indent=4)

    checkpoints = pd.DataFrame(columns=[*epoch_args_key, *checkpoints_key])
    checkpoints.index.name = "epoch"

    if args.load_epoch > 0:
        epoch_dir = out_dir/f"{args.load_epoch}"
        model.load_state_dict(torch.load(epoch_dir/"model.pth"))
        checkpoints = pd.concat([checkpoints, pd.read_csv(epoch_dir/"checkpoints.csv", index_col="epoch")])
        test_acc, label_est = test()
        print(f"state loaded from epoch {args.load_epoch} - test_acc: {test_acc*100:.2f}%")
        if args.test_only:
            pd.DataFrame(data={"Id": test_real_idx, "Category": label_est}).to_csv(epoch_dir/"test_sample.csv", index=False)
    elif args.test_only:
        raise RuntimeError("args.test_only must be set with args.load_epoch")

    if not args.test_only:
        train_losses = []
        for epoch in range(args.load_epoch + 1, args.num_epochs + 1):
            start_time = time()
            train_acc, train_loss = train()
            time_spend = time() - start_time

            train_losses.append(train_loss)
            if len(train_losses) > 5 and train_losses[-1] > np.mean(train_losses[-6:-1]):
                scheduler.step()

            test_acc, label_est = test()

            print(f"| epoch {epoch:3d} | time: {time_spend:7.2f}s | train_acc: {train_acc*100:5.2f}% " +
                  f"| train_loss: {train_loss:11.6f} | test_acc: {test_acc*100:5.2f}%")
            checkpoints.loc[epoch] = {**{k: v for k, v in zip(checkpoints_key, [time_spend, train_acc, train_loss, test_acc])},
                                      **filter_args(vars(args), epoch_args_key)}

            epoch_dir = out_dir/f"{epoch}"
            epoch_dir.mkdir(exist_ok=True)
            torch.save(model.state_dict(), epoch_dir/"model.pth")
            checkpoints.to_csv(epoch_dir/"checkpoints.csv")
            pd.DataFrame(data={"Id": test_real_idx, "Category": label_est}).to_csv(epoch_dir/"test_sample.csv", index=False)
