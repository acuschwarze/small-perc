import sys, pickle
sys.path.insert(0, "libs")

import os, pickle, csv # import packages for file I/O
import time # package to help keep track of calculation time

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

import scipy
import scipy.stats as sst
from scipy.special import comb
from scipy.integrate import simpson
from scipy.signal import argrelextrema
from random import choice

from libs.utils import *
from libs.finiteTheory import *
from visualizations import *
from libs.utils import *
from robustnessSimulations import *
from performanceMeasures import *
from infiniteTheory import *
from finiteTheory import *

fvals = pickle.load(open('data/fvalues.p', 'rb'))
pvals = pickle.load(open('data/Pvalues.p', 'rb'))


# data = pd.read_pickle("bayesian array")
# nodes = pd.read_pickle("bayesian nodes")

# l=len(data[0])

# nwks_list2 = pd.read_csv("nwks_list2")
# #print(len(nwks_list2))

# # print(nwks_list2.iloc[0])
# cwd = os.getcwd() 
# # print("Current working directory:", cwd) 

# def r_pow(x, n, d):
#     """
#     Compute x to the power of n/d (not reduced to lowest
#     expression) with the correct function real domains.
    
#     ARGS:
#         x (int,float,array): base
#         n (int)            : exponent numerator
#         d (int)            : exponent denominator
        
#     RETURNS:
#         x to the power of n/d
#     """
    
#     # list to array
#     if type(x) == list:
#         x = np.array(x)
#     # check inputs
#     if type(n) != int or type(d) != int:
#         raise Exception("Exponent numerator and denominator must be integers")
#     # if denominator is zero
#     if not d:
#         raise Exception("Exponent denominator cannot be 0")
        
#     # raise x to power of n
#     X = x**n
#     # even denominator
#     if not d % 2:
#         # domain is for X>=0 only
#         if type(x) == np.ndarray:
#             X[X<0] = np.nan
#         elif X < 0:
#             X = np.nan
#         res = np.power(X, 1./d)
#         return res
#     # odd denominator
#     else:
#         # domain is all R
#         res = np.power(np.abs(X), 1./d)
#         res *= np.sign(X)
#         return res
    

# roots = [r_pow(data[0][i], 1, int(nodes[0][i])) for i in range(l)] 

# #plt.plot(nodes[0], roots, lw=0, marker='x')
# top20 = np.array([[nodes[0][i], roots[i]] for i in range(l) if roots[i]>=np.percentile(roots,80)]).T
# #plt.plot(top20[0], top20[1], lw=0, marker='x')
# bottom20 = np.array([[nodes[0][i], roots[i]] for i in range(l) if roots[i]<np.percentile(roots,20)]).T
# #plt.plot(bottom20[0], bottom20[1], lw=0, marker='x')

# bottom20_i = np.array([i for i in range(l) if roots[i]<np.percentile(roots,20)]).T
# top20_i = np.array([i for i in range(l) if roots[i]>=np.percentile(roots,80)]).T

# print(bottom20_i)
# print(top20_i)



# def nodecount_edge(file_name = ""):
#     file = open(file_name, "r")
#     #content = file.readlines()
#     content = (line.rstrip() for line in file)  # All lines including the blank ones
#     content = list(line for line in content if line)
#     #print(content)
#     node_list = []
#     edge_list = np.empty(len(content), dtype=object)
#     for i in range(len(content)):
#         edge = content[i].strip()
#         edge = edge.split(" ")
#         edge_list[i] = np.zeros(2)
#         #print("i", i)
#         #print("edge[0]",edge[0])
#         edge_list[i][0] = int(edge[0])
#         edge_list[i][1] = int(edge[1])
#         for j in range(2):
#             node_list.append(int(edge[j]))
#     if 0 in node_list:
#         n = max(node_list) + 1
#     else:
#         n = max(node_list)
#     return n


# fullData = pd.read_csv("fullData.csv")
# #counter_100 = 0

# y_array_b = []
# y_array_t = []
# bottom_nodes = []
# top_nodes = []

# import os

# indices = pd.read_pickle("bayesian indices")
# #print("indices")
# #print(indices)
# #print(indices[0])
# #print('iloc')
# #print(indices.iloc[0][0])
 

# nwks_100 = []
# for i in range(len(nwks_list2)):
#     if nwks_list2.iloc[i][1] == 100:
#         nwks_100.append(i)

# print(nwks_100)

# counter_100 = 0
# for i_b in range(len(bottom20_i)):
#     #counter_100 = 0
#     # for j in nwks_100:
#     #     if j < i_b:
#     #         counter_100 += 1
#     #new_ib = indices.iloc[bottom20_i[i_b]-counter_100][0]
#     new_ib = bottom20_i[i_b]
#     #print("file1",nwks_list2.iloc[i_b][1])
#     file = nwks_list2.iloc[new_ib][1]
#     if nodecount_edge(file_name = file) == 100:
#         counter_100 += 1
#         print("counter", counter_100)
#     elif nodecount_edge(file_name = file) < 100:
#         #file2 = file.replace("C:\\Users\\jj\Downloads\\GitHub\small-perc\\pholme_networks", '')
#         #print("file2",file)
#         #file = file[60:]
#         #print("file3",file2)
#         bottom_nodes.append(nodecount_edge(file_name=file))
#         #values = mega_file_reader2(removal = "random", adj_list = [file])
#     #   print(file)
#     #   print("val",values)
#     #   print("averaged data",values[0])
#     #   print("fin", values[1])
#         #print("nodes")
#         #print(fullData.iloc[bottom20_i[i_b]-counter_100][1])
#         #print(nodecount_edge(file_name=file))

#         if fullData.iloc[bottom20_i[i_b]-counter_100][1] != nodecount_edge(file_name=file):
#             print("nodes")
#             print(fullData.iloc[bottom20_i[i_b]-counter_100][1])
#             print(nodecount_edge(file_name=file))
#         sim = string2array(fullData.iloc[bottom20_i[i_b]-counter_100][3], sep=" ")
#         fin = string2array(fullData.iloc[bottom20_i[i_b]-counter_100][5], sep=" ")
#         y = ((fin - sim) ** 2).mean()
#         if y < 100:
#             y_array_b.append(y)
#             bottom_nodes.append(nodecount_edge(file_name=file))

# y_array_bdf = pd.DataFrame(y_array_b)
# y_array_bdf.to_pickle("bayesian mse bottom 20 (n=100)")

# #counter_100 = 0
# for i_t in range(len(top20_i)):
#     #counter_100 = 0
#     #print("counter", counter_100)
#     # for j in nwks_100:
#     #     if j < i_b:
#     #         counter_100 += 1
#     #new_it = indices.iloc[top20_i[i_t]-counter_100][0]
#     new_it = top20_i[i_t]
#     file = nwks_list2.iloc[new_it][1]
#     if nodecount_edge(file_name = file) >= 100:
#         counter_100 += 1
#     elif nodecount_edge(file_name = file) < 100:
#       sim = string2array(fullData.iloc[top20_i[i_t]-counter_100][3], sep=" ")
#       fin = string2array(fullData.iloc[top20_i[i_t]-counter_100][5], sep=" ")

#       if fullData.iloc[top20_i[i_t]-counter_100][1] != nodecount_edge(file_name=file):
#           print("nodes")
#           print(fullData.iloc[top20_i[i_t]-counter_100][1])
#           print(nodecount_edge(file_name=file))
#       y = ((fin - sim) ** 2).mean()
#       if y < 100:
#         y_array_t.append(y)
#         top_nodes.append(nodecount_edge(file_name=file))

# y_array_tdf = pd.DataFrame(y_array_t)
# y_array_tdf.to_pickle("bayesian mse top 20 (n=100)")

# bottom_nodesdf = pd.DataFrame(bottom_nodes)
# bottom_nodesdf.to_pickle("bottom nodes (n=100)")

# top_nodesdf = pd.DataFrame(top_nodes)
# top_nodesdf.to_pickle("top nodes (n=100)")

# #print("total indices",len(bottom_nodes)+len(top_nodes))

# # top_indices = pd.read_pickle("bayesian mse top 20")
# # bottom_indices = pd.read_pickle("bayesian mse bottom 20")

# # bottom = pd.read_pickle("bottom nodes")
# # top = pd.read_pickle('top nodes')

# # xb = np.zeros(len(bottom))
# # xt = np.zeros(len(top))

# # print("lenbottom",len(bottom))
# # for k in range(len(bottom)):  
# #   print(bottom.iloc[k][0])
# #   xb[k] = bottom.iloc[k][0]

# # for l in range(len(top)):
# #     xt[l] = top.iloc[l][0]



# # top_y = np.zeros(len(top_indices))
# # bot_y = np.zeros(len(bottom_indices))


# # print('bot')
# # print(bottom_indices.iloc[0][0])

# # for i in range(len(top_indices)):
# #   top_y[i] = top_indices.iloc[i][0]


# # for j in range(len(bottom_indices)):
# #   bot_y[j] = bottom_indices.iloc[j][0]

# # print("boty")
# # print(bot_y)




# plt.plot(top_nodes,y_array_t,'x',color= "red", label="top 20")
# plt.plot(bottom_nodes,y_array_b,'x',color= "blue", label="bottom 20")
# plt.xlabel("nodes")
# plt.ylabel('mse')
# plt.legend()
# plt.show()

# mean_top = np.mean(y_array_t)
# print(mean_top)
# mean_bottom = np.mean(y_array_b)
# print(mean_bottom)


def nodecount_edge(file_name = ""):
    #print(file_name)
    file = open(file_name, "r")
    #content = file.readlines()
    content = (line.rstrip() for line in file)  # All lines including the blank ones
    content = list(line for line in content if line)
    if len(content) == 0:
        return 0
    #print(content)
    node_list = []
    edge_list = np.empty(len(content), dtype=object)
    for i in range(len(content)):
        edge = content[i].strip()
        edge = edge.split(" ")
        edge_list[i] = np.zeros(2)
        #print("i", i)
        #print("edge[0]",edge[0])
        edge_list[i][0] = int(edge[0])
        edge_list[i][1] = int(edge[1])
        for j in range(2):
            node_list.append(int(edge[j]))
    if 0 in node_list:
        n = max(node_list) + 1
    else:
        n = max(node_list)
    return n



fullData = pd.read_csv("fullData.csv")
nwks_list2 = pd.read_csv("nwks_list2")
cwd = os.getcwd()


nwks_names = []
data_names = []


for i in range(len(nwks_list2)):
    pathname = nwks_list2.iloc[i][1]
    nwks_listname = os.path.basename(nwks_list2.iloc[i][1])
    if nodecount_edge(file_name = pathname) <= 100:
        nwks_names.append(nwks_listname)

for j in range(len(fullData)):
    data_name = fullDataname = fullData.iloc[j][0]
    data_names.append(data_name)

missing = []

for i in nwks_names:
    if i not in data_names:
        missing.append(i)


for j in data_names:
    if j not in nwks_names:
        missing.append(j)


print(missing)


# datacounter = 0

# for i in range(len(nwks_list2)):
#     nwks_listname = os.path.basename(nwks_list2.iloc[i][1])
#     #print("nwk",nwks_listname)
#     fullDataname = fullData.iloc[datacounter][0]
#     #print("data",fullDataname)
#     if nwks_listname != fullDataname:
#         print("i",i)
#         print("nwk",nwks_listname)
#         datacounter -= 1
#     datacounter += 1


