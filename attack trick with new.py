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

def expectedNthLargestDegree(n, p, N):
    '''Calculate expected value of the degree of the node that has the Nth 
    largest degree in an Erdos--Renyi graph with n nodes and edge probability
    p.

    Parameters
    ----------
    n : int
       Number of nodes.

    N : int
       Number of the node of interest in the degree-ranked (largest to 
       smallest) node sequence.

    p : float
       Edge probability in Erdos Renyi graph.

    Returns
    -------
    mean_Nk (float)
       The expected value of the degree of the node with the Nth largest 
       degree.
    '''

    if N > n:
        print('Cannot find the {}th largest element in a sequence of only {} numbers.'.format(N,n))

    if n in [0, 1] or p == 0:
        return 0

    if n == 2:
        return p

    # probability that a node has at least degree k
    probs_at_least_k = np.cumsum([binomialDistribution.pmf(k, n - 1, p) 
        for k in range(n)][::-1])[::-1]
    # probability that at least N nodes have degree k or larger
    probs_at_least_N_nodes = [1 - binomialDistribution.cdf(N-1, n, probs_at_least_k[k]) 
        for k in range(n)]
    if np.isnan(np.sum(probs_at_least_N_nodes)):
        print('probs_at_least_{}_nodes'.format(N), probs_at_least_N_nodes)
    probs_at_least_N_nodes = np.concatenate([probs_at_least_N_nodes, [0]])
    probs_Nk = probs_at_least_N_nodes[:-1] - probs_at_least_N_nodes[1:]
    mean_Nk = np.sum([probs_Nk[k] * k for k in range(n)])

    return mean_Nk

fvals = pickle.load(open('data/fvalues.p', 'rb'))
pvals = pickle.load(open('data/Pvalues.p', 'rb'))

def attack_trick(p, nodes_array=[10, 20, 30, 40, 50, 60], removal=["attack"], trials = 100):

    if removal == ["random"]:
        remove_bool = False
    elif removal == ["attack"]:
        remove_bool = True
    prob_array = np.zeros(len(nodes_array))
    colors = ["red", "magenta", "purple", "navy"]

    prob_array[:] = p

    for j in range(len(nodes_array)):

        n = nodes_array[j]
        rlcc = np.zeros(n)
        rlcc2 = np.zeros(n)
        graph_array = np.zeros(trials, dtype = object)
        graph_array2 = np.zeros(trials, dtype = object)
        for r in range(trials):
            graph_array[r] = nx.gnp_random_graph(n,p)
            graph_array2[r] = nx.gnp_random_graph(n,p)
        removed_max = np.zeros(n)

        current_p = p
        print(j, 'updated finite')
        for i_r in range(n):
            new_nodes = n-i_r
            max_d = expectedNthLargestDegree(n, p, i_r+1)  - i_r*p
            removed_max[i_r] = max_d
            print("max_d",max_d)
            print("inputs", current_p, new_nodes)

            if np.isnan(current_p):
                rlcc[i_r] = np.nan
            else:
                rlcc[i_r] = calculate_S(current_p, new_nodes, fvals, pvals,lcc_method = "pmult", executable_path=r'C:\Users\f00689q\My Drive\jupyter\small-perc\libs\p-recursion.exe')/(n-i_r)
            if new_nodes <=2:
                current_p = 0

            else:
                current_p = current_p * new_nodes / (new_nodes - 2) - 2 * max_d / ((new_nodes - 1) * (new_nodes - 2))
                current_p = max([current_p, 0])

        # now with different ensemble
        current_p = p
        print(j, 'trick old')
        for i_r in range(n):
            new_nodes = n-i_r
            max_d = 0
            for graph in graph_array2:
                v = sorted(graph.degree, key=lambda x: x[1], reverse=True)[0][0]
                max_d += graph.degree(v)
                graph.remove_node(v)
            max_d /= trials
            removed_max[i_r] = max_d
            #print("max_d",max_d)
            rlcc2[i_r] = calculate_S(current_p, new_nodes, fvals, pvals,lcc_method = "pmult", executable_path=r'C:\Users\f00689q\My Drive\jupyter\small-perc\libs\p-recursion.exe')/(n-i_r)
            if new_nodes <=2:
                current_p = 0

            else:
                current_p = current_p * new_nodes / (new_nodes - 2) - 2 * max_d / ((new_nodes - 1) * (new_nodes - 2))
                current_p = max([current_p, 0])
        
        print(j, 'sim')
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
        plt.plot(np.arange(n) / n, rlcc2,
                    label="trick2 n: " + str(n), color=colors[j])
        plt.plot(np.arange(n) / n,
                     finiteTheory.relSCurve(prob_array[j], n,
                                            attack=remove_bool, fdict=fvals, pdict=pvals, lcc_method_relS="pmult", executable_path='libs/p-recursion-float128.exe'),
                     label="fin n: " + str(n), color=colors[j], linestyle = "--")
        plt.plot(np.arange(n) / n, rlcc,
                    label="updated fin: " + str(n), color=colors[j], linestyle=':')

    ax.legend(bbox_to_anchor=(1, 0.5))
    plt.xlabel('number of nodes left')
    plt.ylabel('S')
    plt.subplots_adjust(right=0.7)
    plt.savefig('attack_trick_new_p{}.png'.format(p))
    plt.clf()

for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
    attack_trick(p, nodes_array=[5, 10, 20], removal=["attack"], trials=500)