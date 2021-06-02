#! /usr/bin/env python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import matplotlib.pyplot as plt
import numpy as np

from util import *


# Load data
print("Loading data...")
graphs, num_classes, label_map, _, graph_name_map = load_cached_data("KAGGLE")

def collab_classifier(idx):
    return 1 if idx <= 2600 else 2 if idx <= 3375 else 3

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
    degree_list = np.array(degree_list)
    avg_degree = np.mean(degree_list)
    var_degree = np.var(degree_list)
    return avg_degree, var_degree

f_3 = open("/home/jeha/DL/NoreGraph/Graph-Transformer/U2GNN_pytorch/filtered_graph_3.txt",'w')
f_2 = open("/home/jeha/DL/NoreGraph/Graph-Transformer/U2GNN_pytorch/filtered_graph_not2.txt", 'w')
filtered_3 = []
filtered_2 = []

for graph in graphs:
    avg_degree, var_degree = _post_processing(graph)
    if len(graph.g) >= 239 or avg_degree >= 61:
        filtered_3.append(graph.name)
    if avg_degree > 25 :
        filtered_2.append(graph.name)

            
f_2.write(",".join(map(str,filtered_2)))  
f_3.write(",".join(map(str,filtered_3)))
f_3.close()    
f_2.close()    
