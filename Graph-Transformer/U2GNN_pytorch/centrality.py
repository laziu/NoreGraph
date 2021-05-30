#! /usr/bin/env python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

import numpy as np
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
args = parser.parse_args()

print(args)

# Load data
print("Loading data...")
graphs, num_classes, label_map, _, graph_name_map = load_cached_data(args.dataset)


# graphs = g_list
temp = []
b_cent = open(current_root/'b_c.txt','w')
c_cent = open(current_root/'c_c.txt','w')
d_cent = open(current_root/'d_c.txt','w')
weird_cent = open(current_root/'centrality_result/weird_cent.txt','w')
d_list = []
c_list = []
b_list = []
num_graph = 0
for i, g in enumerate(graphs):
    #temp = b_cent.readline().split(',')
    temp = list(nx.betweenness_centrality(g.g).values())
    b_cent.write(','.join(map(str,temp)))
    b_cent.write('\n')
    if max(map(float,temp)) < 0.00001 :
        b_list.append(g.name)

    #temp = c_cent.readline().split(',')
    temp = list(nx.closeness_centrality(g.g).values())
    c_cent.write(','.join(map(str,temp)))
    c_cent.write('\n')
    if min(map(float,temp)) > 0.99 :
        c_list.append(g.name)
    
    #temp = d_cent.readline().split(',')
    temp = list(nx.degree_centrality(g.g).values())
    d_cent.write(','.join(map(str,temp)))
    d_cent.write('\n')  
    if min(map(float,temp)) > 0.99 :
        d_list.append(g.name)

      
intersection_list = list(set(b_list) & set(c_list) & set(d_list))
weird_cent.write(','.join(map(str,intersection_list)))
print(len(intersection_list))
print(intersection_list)

print(len(b_list))
print(len(c_list))
print(len(d_list))

b_cent.close()
c_cent.close()
d_cent.close()
weird_cent.close()