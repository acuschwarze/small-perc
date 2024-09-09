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


def one_perc_thresh_table(threshold=.5, nodes=[10, 20, 30, 40, 50, 60], removal=["attack"]):
    one_perc_table = np.zeros((len(nodes), 4), dtype=object)

    if removal == ["random"]:
        remove_bool = False
    elif removal == ["attack"]:
        remove_bool = True
    percthresh = threshold
    nodes_array = nodes
    prob_array = np.zeros(len(nodes_array))
    colors = ["red", "orange", "yellow", "green", "purple", "magenta", "cyan"]

    for i in range(len(nodes_array)):
        prob_array[i] = .8
        #prob_array[i] = 1 / (percthresh * (nodes_array[i] - 1))

    fig = plot_graphs(numbers_of_nodes=[nodes_array[0]], edge_probabilities=[prob_array[0]],
                     graph_types=['ER'], remove_strategies=removal,
                     performance='relative LCC', num_trials=100,
                     smooth_end=False, forbidden_values=[], fdict=fvals, lcc_method_main="pmult", savefig='',
                     simbool=True)

    for j in range(len(nodes_array)):
        sim_data = completeRCData(numbers_of_nodes=[nodes_array[j]],
                                  edge_probabilities=[prob_array[j]], num_trials=1000,
                                  performance='relative LCC', graph_types=['ER'],
                                  remove_strategies=removal)
        data_array = np.array(sim_data[0][0][0][0])

        # exclude the first row, because it is the number of nodes
        data_array = data_array[1:]

        # this can prevent some sort of bug about invalid values
        for val in []:
            data_array[data_array == val] = np.nan

        # plot simulated data
        removed_fraction = np.arange(nodes_array[j]) / nodes_array[j]
        line_data = np.nanmean(data_array, axis=0)

        one_perc_table[j][0] = nodes_array[j]
        one_perc_table[j][1] = prob_array[j]
        one_perc_table[j][2] = line_data
        one_perc_table[j][3] = finiteTheory.relSCurve(prob_array[j], nodes_array[j],
                                        attack=remove_bool, fdict=fvals, pdict=pvals, lcc_method_relS="pmult", executable_path='libs/p-recursion-float128.exe')
        if j != 0:
            plt.plot(removed_fraction, line_data,
                     'o', label="n={} , p={}".format(nodes_array[j], prob_array[j]), color=colors[j])
            plt.plot(np.arange(nodes_array[j]) / nodes_array[j],
                     finiteTheory.relSCurve(prob_array[j], nodes_array[j],
                                            attack=remove_bool, fdict=fvals, pdict=pvals, lcc_method_relS="pmult", executable_path='libs/p-recursion-float128.exe'),
                     label="n: " + str(nodes_array[j]), color=colors[j])

    plt.legend()
    plt.show()
    #fig.savefig("percolation_graph"+str(percthresh)+ ".png")
    
    # df = pd.DataFrame(one_perc_table)
    # df.columns = ["nodes", "prob", "simulated RLCC", "fin theory RLCC"]
    # return df


# n = int(sys.argv[1])
# threshold = float(sys.argv[2])
# r = sys.argv[3]

# df = one_perc_thresh_table(threshold=threshold, 
#                            nodes=[n], #[10, 15, 25, 50, 75, 100], 
#                            removal=[r])

# df.to_pickle('percolation_{}_n{}_t{}.p'.format(r, n, threshold))


# colors = ['red','blue','orange','green','purple','cyan']
# fig = plt.figure(figsize=(8, 8))
# perc_rand = pd.read_pickle("percolation_random")
# nodes = perc_rand.nodes.values
# data = perc_rand.loc[:,"fin theory RLCC"]
# sim = perc_rand.loc[:,'simulated RLCC']
# n = len(perc_rand.nodes.values)
# for j in range(n):
#     nodes_array = np.arange(nodes[j], dtype=float)/nodes[j]
#     plt.plot(nodes_array, data[j], color = colors[j])
#     plt.plot(nodes_array,sim[j],'o',color = colors[j])
# p = 1/(.2*(100-1))
# plt.plot(nodes_array, infiniteTheory.relSCurve(100, p,
#                             attack=False, smooth_end=False), label = "inf theory")
# plt.xlabel("percent nodes removed")
# plt.ylabel("relative LCC")
# plt.title("relative LCC over nodes removed randomly")
# plt.savefig("one_perc_graph_random")




#8/12
# colors = ['red','blue','orange','green','purple','cyan']
# fig = plt.figure(figsize=(8, 8))

# pa_10 = pd.read_pickle("percolation_attack_n10_t0.4.p" )
# pa_15 = pd.read_pickle("percolation_attack_n15_t0.4.p" )
# pa_25 = pd.read_pickle("percolation_attack_n25_t0.4.p")
# pa_50 = pd.read_pickle("percolation_attack_n50_t0.4.p" )
# #pa_75 = pd.read_pickle("percolation_attack_n75_t0.4.p" )
# #pa_100 = pd.read_pickle("percolation_attack_n100_t0.4.p" )
# list_files = [pa_10,pa_15,pa_25,pa_50]

# for i in range(len(list_files)):
#     n = list_files[i].iloc[0][0]
#     nodes_array = np.arange(n) / n
#     sim = list_files[i].iloc[0][2]
#     fin = list_files[i].iloc[0][3]
#     plt.plot(nodes_array, fin, color = colors[i])
#     plt.plot(nodes_array, sim,'o',color = colors[i])

# n = 50
# p = 1/(.4*(n-1))
# nodes_array = np.arange(n) / n
# plt.plot(nodes_array, infiniteTheory.relSCurve(n, p,
#                             attack=True, smooth_end=False), label = "inf theory")
# plt.xlabel("percent nodes removed")
# plt.ylabel("relative LCC")
# plt.title("relative LCC over nodes removed targeted")
# plt.savefig("one_perc_graph_attack_pointfour")


# p is .8 right now (in the function)
one_perc_thresh_table(threshold=.2, nodes=[6], removal=["attack"])