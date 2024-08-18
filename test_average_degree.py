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

################################################################################

num_trials = 200
remove_nodes = 'attack'

################################################################################

labels = [r'$c_{sim}$', r'$c_{epATA}$', r'$c_{sim+epATA}$', r'$K_{sim}$', r'$K_{from current}$', r'$K_{iteration}$', r'$K_{from first}$']
colors = ['navy', 'mediumblue', 'lightblue', 'red', 'tomato', 'darkorange', 'orange']

################################################################################

for n in [5,10,20]:
    data_array = np.zeros((num_trials, 8, n), dtype=float)
    data_array[0] = np.arange(n)

    for ip, p in enumerate([0.1, 0.5, 0.9]):
        
        for j in range(num_trials):

            g = sampleNetwork(n, p, graph_type='ER')
            c = averageDegree(g)
            p_new = p
            
            for i in range(n):
                    # calculate true mean degree
                    data_array[j, 1, i] = averageDegree(g) 
                    # calculate mean degree according to `edgeProbabilityAfterTargetedAttack`
                    p_next = edgeProbabilityAfterTargetedAttack(n-i, p_new)
                    data_array[j, 2, i] = 2*p_new*binom(n-i,2)/n
                    # calculate mean degree of a new G(n,p) graph with the predicted p
                    g2 = sampleNetwork(n-i, p_new, graph_type='ER')
                    data_array[j, 3, i] = averageDegree(g2) 

                    # calculate true max degree ("sim")
                    data_array[j, 4, i] = np.max([d for n, d in g.degree()] )
                    # calculate max degree with independence assumption: ("from current")
                    # we calculate the EMD of a graph with the same n and p as the current graph 
                    if g.number_of_edges():
                        data_array[j, 5, i] = expectedMaxDegree(
                            g.number_of_nodes(), g.number_of_edges()/binom(n-i,2)) 
                    # calculate max degree with independence assumption + ER assumption: ("from iteration")
                    # we calculate the EMD from the predicted p_new
                    
                    data_array[j, 6, i] = expectedMaxDegree(
                        g.number_of_nodes(), p_new)                     
                    #print('(n, p_new, emd) = ({},{},{})'.format(n, p_new, data_array[j, 6, i]))

                    # calculate max degree with independence assumption but no ER assumption: ("from first")
                    # we calculate the EMD from the Nth largest degree in original network
                    if i <= n:
                        data_array[j, 7, i] = expectedNthLargestDegree(n,  p, i+1) 

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

        plt.subplot(1,3,1+ip)
        plt.title('p={}'.format(p))
        for l in range(4,len(data_array[0])): #range(1,len(data_array[0])):
            plt.plot(np.nanmean(data_array[:,l],axis=0), marker='ovsd'[l%4], fillstyle='none',
                ls=['--','-'][l//4], lw=0, c=colors[l-1], label=labels[l-1])
        #print(np.nanmean(data_array, axis=0))
    plt.legend()
    plt.savefig('test_average_degree_n{}.png'.format(n))
    plt.clf()
