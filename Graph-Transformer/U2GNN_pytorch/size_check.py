#! /usr/bin/env python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
torch.manual_seed(123)

import numpy as np
np.random.seed(123)
import time

from pytorch_U2GNN_UnSup import *
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from scipy.sparse import coo_matrix
from util import *
from sklearn.linear_model import LogisticRegression
import networkx as nx
import statistics
import json
import pickle
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)

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
args = parser.parse_args()

print(args)

# Load data
print("Loading data...")
graphs, num_classes, label_map, _, graph_name_map = load_cached_data(args.dataset)

max=0
min=500
for graph in graphs:
    if max < len(graph.g):
        max = len(graph.g)
    if min > len(graph.g):
        min = len(graph.g)
print(max, min)
""" f_index = open('/root/DLProject/NoreGraph/Graph-Transformer/U2GNN_pytorch/centrality_result/weird_cent.txt','r')

index = list(map(int, f_index.readline().rstrip('\n').split(',')))
#print(index)
size_1 = []
size_2 = []
size_3 = []
for graph in graphs:
    if graph.name in index:
        print('{}th / size {} / label {}'.format(graph.name , len(graph.g), graph.label))
        if graph.label == 1:
            size_1.append(len(graph.g))
        elif graph.label == 2:
            size_2.append(len(graph.g))

print(min(size_1))
print(max(size_1))

print(min(size_1))
print(max(size_2))

bins = np.arange(32,239,3)
plt.hist(size_2,bins=bins)

plt.hist(size_1,bins=bins)
plt.show()
plt.savefig('size.png') """