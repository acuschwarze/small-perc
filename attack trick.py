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

def attack_trick(p, nodes_array=[10, 20, 30, 40, 50, 60], removal=["attack"], trials = 100):

    if removal == ["random"]:
        remove_bool = False
    elif removal == ["attack"]:
        remove_bool = True
    prob_array = np.zeros(len(nodes_array))
    colors = ["red", "magenta", "purple", "navy"]

    for i in range(len(nodes_array)):
        prob_array[i] = p
        #prob_array[i] = 1 / (percthresh * (nodes_array[i] - 1))

    # fig = plot_graphs(numbers_of_nodes=[nodes_array[0]], edge_probabilities=[prob_array[0]],
    #                  graph_types=['ER'], remove_strategies=removal,
    #                  performance='relative LCC', num_trials=100,
    #                  smooth_end=False, forbidden_values=[], fdict=fvals, lcc_method_main="pmult", savefig='',
    #                  simbool=True)

    for j in range(len(nodes_array)):
        n=nodes_array[j]
        rlcc = np.zeros(n)
        rlcc2 = np.zeros(n)
        graph_array = np.zeros(trials, dtype = object)
        graph_array2 = np.zeros(trials, dtype = object)
        for r in range(trials):
            graph_array[r] = nx.gnp_random_graph(n,p)
            graph_array2[r] = nx.gnp_random_graph(n,p)
        removed_max = np.zeros(n)

        current_p = p
        for i_r in range(n):
            new_nodes = n-i_r
            max_d = 0
            for graph in graph_array:
                v = sorted(graph.degree, key=lambda x: x[1], reverse=True)[0][0]
                max_d += graph.degree(v)
                graph.remove_node(v)
            max_d /= trials
            removed_max[i_r] = max_d
            print("max_d",max_d)
            rlcc[i_r] = calculate_S(current_p, new_nodes, fvals, pvals,lcc_method = "pmult", executable_path=r'C:\Users\f00689q\My Drive\jupyter\small-perc\libs\p-recursion.exe')/(n-i_r)
            if new_nodes <=2:
                current_p = 0

            else:
                current_p = current_p * new_nodes / (new_nodes - 2) - 2 * max_d / ((new_nodes - 1) * (new_nodes - 2))
                current_p = max([current_p, 0])

        # now with different ensemble
        current_p = p
        for i_r in range(n):
            new_nodes = n-i_r
            max_d = 0
            for graph in graph_array2:
                v = sorted(graph.degree, key=lambda x: x[1], reverse=True)[0][0]
                max_d += graph.degree(v)
                graph.remove_node(v)
            max_d /= trials
            removed_max[i_r] = max_d
            print("max_d",max_d)
            rlcc2[i_r] = calculate_S(current_p, new_nodes, fvals, pvals,lcc_method = "pmult", executable_path=r'C:\Users\f00689q\My Drive\jupyter\small-perc\libs\p-recursion.exe')/(n-i_r)
            if new_nodes <=2:
                current_p = 0

            else:
                current_p = current_p * new_nodes / (new_nodes - 2) - 2 * max_d / ((new_nodes - 1) * (new_nodes - 2))
                current_p = max([current_p, 0])
        
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
        
        ax = plt.subplot(111)
        plt.plot(removed_fraction, line_data,
                     'o', label="n={} , p={}".format(nodes_array[j], prob_array[j]), color=colors[j])
        plt.plot(np.arange(n) / n, rlcc,
                    label="trick n: " + str(n), color=colors[j], linestyle=':')
        plt.plot(np.arange(n) / n, rlcc2,
                    label="trick2 n: " + str(n), color=colors[j])
        plt.plot(np.arange(n) / n,
                     finiteTheory.relSCurve(prob_array[j], n,
                                            attack=remove_bool, fdict=fvals, pdict=pvals, lcc_method_relS="pmult", executable_path='libs/p-recursion-float128.exe'),
                     label="fin n: " + str(n), color=colors[j], linestyle = "--")

    ax.legend(bbox_to_anchor=(1, 0.5))
    plt.xlabel('number of nodes left')
    plt.ylabel('S')
    plt.subplots_adjust(right=0.7)
    plt.savefig('attack_trick_p{}.png'.format(p))
    plt.clf()

for p in [0.9]: #n [0.1, 0.3, 0.5, 0.7, 0.9]:
    attack_trick(p, nodes_array=[20], removal=["attack"], trials=100)