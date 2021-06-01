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



max_l=0
min_l=500
for graph in graphs:
    if max_l < len(graph.g):
        max_l = len(graph.g)
    if min_l > len(graph.g):
        min_l = len(graph.g)


degree_1 = []
degree_2 = []
degree_3 = []

var_1 = []
var_2 = []
var_3 = []

size_1 = []
size_2 = []
size_3 = []

for graph in graphs:
    avg_degree, var_degree = _post_processing(graph)
    # print('{}th / size {} / label {}'.format(graph.name , len(graph.g), graph.label))
    if graph.label == 1:
        size_1.append(len(graph.g))
        degree_1.append(avg_degree)
        var_1.append(var_degree)
    elif graph.label == 2:
        size_2.append(len(graph.g))
        degree_2.append(avg_degree)
        var_2.append(var_degree)
    elif graph.label == 0:
        size_3.append(len(graph.g))
        degree_3.append(avg_degree)
        var_3.append(var_degree)


max_degree = max(max(degree_1), max(degree_2), max(degree_3))
min_degree = min(min(degree_1), min(degree_2), min(degree_3))

max_var = max(max(var_1), max(var_2), max(var_3))
min_var = min(min(var_1), min(var_2), min(var_3))

print(max_var, min_var)


# drawing size chart
""" 
bins = np.arange(32,492,5)
plt.title("# of graphs vs # of nodes (log scale)")
plt.hist(size_1, bins=bins, label ='label = 1', log = True)
plt.hist(size_2, bins=bins, label ='label = 2', log = True)
plt.hist(size_3, bins=bins, label ='label = 3', log = True)
plt.legend(loc='upper right')
plt.xlabel("# of nodes in a graph")
plt.ylabel("# of graphs")
plt.show()
plt.savefig('size_all.png') 
 """
 
# drawing degree chart
"""  
bins = np.arange(min_degree,max_degree,5)
plt.title("# of graphs vs avg. degree of a graph")
plt.hist(degree_1, bins=bins, label ='label = 1')
plt.hist(degree_2, bins=bins, label ='label = 2')
plt.hist(degree_3, bins=bins, label ='label = 3')
plt.legend(loc='upper right')
plt.xlabel("avg. degree of a graph")
plt.ylabel("# of graphs")
plt.show()
plt.savefig('degree_all.png') 
 """

# drawing var(degree) chart

 
bins = np.arange(min_var,max_var,5)
plt.title("# of graphs vs degree variance of a graph")
plt.hist(var_1, bins=bins, label ='label = 1')
plt.hist(var_2, bins=bins, label ='label = 2')
plt.hist(var_3, bins=bins, label ='label = 3')
plt.legend(loc='upper right')
plt.xlabel("degree variance of a graph")
plt.ylabel("# of graphs")
plt.ylim([0,100])
plt.xlim([0,1500])
plt.show()
plt.savefig('var_all.png') 
