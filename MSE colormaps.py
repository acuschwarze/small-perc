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


def gradient(nodenumber, probsnumber, removal, mse_type):
    nodes = np.arange(1, nodenumber + 1, 1)
    probs = np.linspace(0, 1, num=probsnumber + 1)  # probsnumber = 10 means gaps of .1
    grid_table = np.zeros((nodenumber, probsnumber + 1), dtype=object)
    heatmap_array = np.zeros((probsnumber + 1, nodenumber), dtype=object)

    if removal == "random":
        remove_bool = False
    elif removal == "attack":
        remove_bool = True

    for i_n in range(nodenumber):
        for i_p in range(probsnumber):
            sim_data = completeRCData(numbers_of_nodes=[nodes[i_n]],
                                      edge_probabilities=[probs[i_p]], num_trials=100,
                                      performance='relative LCC', graph_types=['ER'],
                                      remove_strategies=[removal])
            rdata_array = np.array(sim_data[0][0][0][0])
            rdata_array = rdata_array[1:]
            for val in []:
                rdata_array[rdata_array == val] = np.nan

            line_data = np.nanmean(rdata_array, axis=0)

            if mse_type == "simulations":
                heatmap_array[i_p][i_n] = scipy.integrate.simpson(line_data, dx=1 / (nodes[i_n] - 1))
                grid_table[i_n][i_p] = line_data
            elif mse_type == "finite":
                fin_curve = finiteTheory.relSCurve(probs[i_p], nodes[i_n],
                                                   attack=remove_bool, fdict=fvals, pdict=pvals,
                                                   lcc_method_relS="pmult")
                heatmap_array[i_p][i_n] = ((fin_curve - line_data) ** 2).mean()
                grid_table[i_n][i_p] = np.zeros(2,dtype=object)
                grid_table[i_n][i_p][0] = line_data
                grid_table[i_n][i_p][1] = fin_curve
            elif mse_type == "infinite":
                inf_curve = infiniteTheory.relSCurve(nodes[i_n], probs[i_p],
                                                     attack=remove_bool, smooth_end=False)
                heatmap_array[i_p][i_n] = ((inf_curve - line_data) ** 2).mean()
                grid_table[i_n][i_p] = np.zeros(2,dtype=object)
                grid_table[i_n][i_p][0] = line_data
                grid_table[i_n][i_p][1] = inf_curve
            elif mse_type == "difference":
                fin_curve = finiteTheory.relSCurve(probs[i_p], nodes[i_n],
                                                   attack=remove_bool, fdict=fvals, pdict=pvals,
                                                   lcc_method_relS="pmult")
                inf_curve = infiniteTheory.relSCurve(nodes[i_n], probs[i_p],
                                                     attack=remove_bool, smooth_end=False)
                heatmap_array[i_p][i_n] = ((inf_curve - line_data) ** 2).mean() - ((fin_curve - line_data) ** 2).mean()
                grid_table[i_n][i_p] = np.zeros(3,dtype=object)
                grid_table[i_n][i_p][0] = line_data
                grid_table[i_n][i_p][1] = fin_curve
                grid_table[i_n][i_p][2] = inf_curve

    heatmap_array = heatmap_array.tolist()

    xnodes, yprobs = np.meshgrid(nodes, probs)

    plt.pcolormesh(xnodes, yprobs, heatmap_array)
    plt.xlabel("nodes")
    plt.ylabel("probability")
    plt.title("heatmap of AUC MSE")
    plt.colorbar()  # need a colorbar to show the intensity scale
    plt.show()
    df = pd.DataFrame(grid_table)
    return df


# output1 = gradient(50, 10, "random", "finite")
# output1.to_pickle("gradient 50n 10p fin")
# output2 = gradient(50, 10, "random", "infinite")
# output2.to_pickle("gradient 50n 10p inf")
# output3 = gradient(50, 10, "random", "difference")
# output3.to_pickle("gradient 50n 10p diff")
#
#
# output1 = gradient(50, 10, "attack", "finite")
# output1.to_pickle("gradient 50n 10p fin")
# output2 = gradient(50, 10, "attack", "infinite")
# output2.to_pickle("gradient 50n 10p inf")
# output3 = gradient(50, 10, "attack", "difference")
# output3.to_pickle("gradient 50n 10p diff")


# graph1 = pd.read_pickle("gradient 50n 10p fin")
# #print(graph1)
# n = len(graph1)
# p = len(graph1.columns)
# # print('n')
# # print(n)
# # print('p')
# # print(p)
# print(graph1.iloc[10])
# heatmap1 = np.zeros((p,n))
# nodes_array = np.arange(1,n+1,1)
# probs_array = np.linspace(0,1,p)
# #print(probs_array)
# print("here")
# print(graph1.iloc[11][0])
# for i_n in range(n):
#     for i_p in range(p):
#         if type(graph1.iloc[i_n][i_p]) == int:
#             break
        
#         # print("start",graph1[i_n][i_p])
#         # print("i_n",i_n)
#         # print("i_p",i_p)
#         fin = graph1.iloc[i_n][i_p][0]
#         sim = graph1.iloc[i_n][i_p][1]
#         heatmap1[i_p][i_n] = ((fin-sim)**2).mean()

# heatmap1 = heatmap1.tolist()
# xnodes, yprobs = np.meshgrid(nodes_array, probs_array)

# plt.pcolormesh(xnodes, yprobs, heatmap1)
# plt.xlabel("nodes")
# plt.ylabel("probability")
# plt.title("heatmap of AUC MSE")
# plt.colorbar()
# plt.savefig("mse finite")

graph2 = pd.read_pickle("gradient 50n 10p inf")
#print(graph1)
n = len(graph2)
p = len(graph2.columns)

heatmap2 = np.zeros((p,n))
nodes_array = np.arange(1,n+1,1)
probs_array = np.linspace(0,1,p)
print(graph2.iloc[11][0])
for i_n in range(n):
    for i_p in range(p):
        if type(graph2.iloc[i_n][i_p]) == int:
            break
        
        # print("start",graph1[i_n][i_p])
        # print("i_n",i_n)
        # print("i_p",i_p)
        fin = graph2.iloc[i_n][i_p][0]
        sim = graph2.iloc[i_n][i_p][1]
        heatmap2[i_p][i_n] = ((fin-sim)**2).mean()

heatmap2 = heatmap2.tolist()
xnodes, yprobs = np.meshgrid(nodes_array, probs_array)

plt.pcolormesh(xnodes, yprobs, heatmap2)
plt.xlabel("nodes")
plt.ylabel("probability")
plt.title("heatmap of AUC MSE")
plt.colorbar()
plt.savefig("mse infinite")