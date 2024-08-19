import sys, pickle
sys.path.insert(0, "libs")

from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

import os, pickle, csv # import packages for file I/O
import time # package to help keep track of calculation time

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

import scipy
import scipy.stats as sst
from scipy.special import comb, binom
from scipy.integrate import simpson
from scipy.signal import argrelextrema
from random import choice

from libs.utils import *
from libs.finiteTheory import *
from libs.performanceMeasures import *
from libs.utils import *

################################################################################

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
    probs_at_least_N_nodes = np.concatenate([probs_at_least_N_nodes, [0]])
    probs_Nk = probs_at_least_N_nodes[:-1] - probs_at_least_N_nodes[1:]
    mean_Nk = np.sum([probs_Nk[k] * k for k in range(n)])

    return mean_Nk

def expectedMaxDegreeWithVariableExponent(n, p, fun=lambda n_, p_:n_):
    '''Calculate expected value of the maximum degree in an Erdos--Renyi graph
    with n nodes and edge probability p.

    Parameters
    ----------
    n : int
       Number of nodes.

    p : float
       Edge probability in Erdos Renyi graph.

    Returns
    -------
    mean_k_max (float)
       The expected value of the maximum degree.
    '''
    if n in [0, 1] or p == 0:
        return 0

    if n == 2:
        return p
        
    #k_max = 0
    probs_k_or_less = np.array([binomialDistribution.cdf(k, n - 1, p) for k in range(n)])
    probs_at_least_k = np.concatenate([[1], np.array(1 - probs_k_or_less[:-1])])
    #probs_at_least_k = np.cumsum([binomialDistribution.pmf(k, n - 1, p) for k in range(n)][::-1])[::-1]
    probs_at_least_one_node = 1 - (1 - probs_at_least_k) ** fun(n,p)

    # every node has at least degree zero
    #probs_at_least_one_node[0] = 1
    # at least one node has degree 1 if the graph is not empty
    #probs_at_least_one_node[1] = 1 - binomialDistribution.pmf(0, n * (n - 1) / 2, p)

    probs_at_least_one_node = np.concatenate([probs_at_least_one_node, [0]])
    probs_kmax = probs_at_least_one_node[:-1] - probs_at_least_one_node[1:]
    mean_k_max = np.sum([probs_kmax[k] * k for k in range(n)])

    return mean_k_max

################################################################################

num_trials = 200
remove_nodes = 'attack'

################################################################################

labels = [r'$K_{sim}$', r'$K_{from current}$', r'$K_{from first}$', r'$K_{from first-ip}$', r'$K_{iterative}$']
colors = ['red', 'navy', 'mediumblue', 'lightblue', 'darkgreen', 'darkorange', 'orange']

################################################################################
prange = [0.9]

for n in [10,20]:
    data_array = np.zeros((num_trials, 6, n), dtype=float)
    data_array[0] = np.arange(n)

    for ip, p in enumerate(prange):
        
        for j in range(num_trials):

            g = sampleNetwork(n, p, graph_type='ER')
            m0 = g.number_of_edges()
            c = averageDegree(g)
            p_new = p
            
            for i in range(n):
                    
                    # calculate mean degree according to `edgeProbabilityAfterTargetedAttack`
                    p_next = edgeProbabilityAfterTargetedAttack(n-i, p_new)

                    # calculate true max degree ("sim")
                    data_array[j, 1, i] = m0 - g.number_of_edges()

                    # calculate max degree with independence assumption: ("from current")
                    # we calculate the EMD of a graph with the same n and p as the current graph 
                    
                    if i < n-1:
                        if g.number_of_edges():
                            edges_removed = expectedMaxDegree(g.number_of_nodes(), g.number_of_edges()/binom(g.number_of_nodes(),2))
                        else:
                            edges_removed = 0

                        data_array[j, 2, i+1] = data_array[j, 2, i] + edges_removed
                        #print('i, data_array[j, 2, i+1], edges_removed', i, data_array[j, 2, i+1], edges_removed)
                        #print('triangles', data_array[j, 2, :])
                        
                    # calculate max degree with independence assumption but no ER assumption: ("from first")
                    # we calculate the EMD from the Nth largest degree in original network
                    if i <= n:
                        if i > 0:
                            data_array[j, 3, i] = data_array[j, 3, i-1] + expectedNthLargestDegree(n,  p, (i-1)+1) 

                    # same as before but with correction for edges to already removed nodes
                    #if i > 0:
                    #    data_array[j, 4, i] = data_array[j, 4, i-1] + data_array[j, 3, i] - p

                    # old method from finite theory
                    if i > 0:
                        data_array[j, 5, i] = data_array[j, 5, i-1] + expectedMaxDegree(
                            g.number_of_nodes()+1, p_new)  

                    p_new = p_next
                    
                    if i == n:
                        break

                    # find a node to remove
                    if remove_nodes == 'random':
                        # select a random node
                        v = choice(list(g.nodes()))
                    elif remove_nodes == 'attack':
                        # select the most connected node
                        v = sorted(g.degree, key=lambda x: x[1], 
                            reverse=True)[0][0]
                    else:
                        raise ValueError(
                            'I dont know that mode of removing nodes')
                        v = None
                    # remove node
                    g.remove_node(v)

        ax = plt.subplot(1,len(prange),1+ip)
        plt.title('p={}'.format(p))
        for l in [1,2,5]: #range(1,2): #len(data_array[0])): 
            print(l)
            plt.plot(np.nanmean(data_array[:,l],axis=0), marker='ovsd'[l%4-1], fillstyle='none',
                lw=0, c=colors[l-1], label=labels[l-1])
        #print(np.nanmean(data_array, axis=0))
    plt.legend()
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    #plt.xticks(np.arange(n))
    #plt.yticks(np.arange(1+np.nanmax(data_array[:,1])))
    plt.grid(which='major', visible=True, c='k')
    plt.grid(which='minor', visible=True, c='gray')
    plt.savefig('test_num_removed_edges_n{}_p{}.pdf'.format(n,p))
    plt.clf()
