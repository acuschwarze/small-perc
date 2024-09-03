import sys, pickle
sys.path.insert(0, "libs")

import os, pickle, csv, itertools # import packages for file I/O
import time # package to help keep track of calculation time

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

import scipy
import scipy.stats as sst
from scipy.stats import binom as binomialDistribution
from scipy.special import comb, binom
from scipy.integrate import simpson
from scipy.signal import argrelextrema
from random import choice
from scipy.interpolate import make_interp_spline
from matplotlib import pyplot as plt

import matplotlib.colors as mcolors

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

    print(n, p, N)

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

def geometric_approach(n, p, i, colors=None, plot=False):

    # set colors
    if colors is None:
        colors = list((mcolors.TABLEAU_COLORS).values())
    
    # Binomial distribution
    x = np.arange(0, n)
    binomial_probs = binomialDistribution.pmf(x, n-1, p)
    if plot: 
        plt.plot(x, binomial_probs, color='gray', lw=0, marker='x')
    
    # Spline interpolation of the binomial distribution
    grid = 1000
    x_spline = np.linspace(0, n, n*grid+1)  # More points for a smooth spline
    spline = make_interp_spline(x, binomial_probs, k=3)
    binomial_spline = spline(x_spline)

    # normalize a binomial spline
    binomial_spline = binomial_spline/np.sum(binomial_spline)*grid

    # intialize max degrees array
    max_degrees = np.zeros(i)
    
    # remove maximum-degree nodes, one at a time
    for j in range(i):

        print('j', j)

        if plot:
            plt.plot(x_spline, binomial_spline, label='removed {}'.format(j), color=colors[j%len(colors)])
            
        # find truncation point
        cumulative_spline = np.cumsum(binomial_spline[::-1])[::-1]
        k_star_index = np.where(cumulative_spline <= 1/n*grid)[0][0]
        k_star = x_spline[k_star_index]
        if plot:
            plt.axvline(x=k_star, color=colors[j%len(colors)], linestyle='--', label=r'$k_j* = {:.3f}$'.format(k_star, j))

    
        # Truncate right at kmax 
        truncated_spline = np.copy(binomial_spline)
        truncated_spline[k_star_index + 1:] = 0  # Truncate at k*
    
        # shift distribution
        shift = k_star / (n - 1)
        mean_k = np.sum(x_spline*binomial_spline)/grid
        c = ((n-j)*k_star-(n-1-j)*mean_k)/(k_star-mean_k)/binom(n-j,2)
        shifts = c*np.arange(len(truncated_spline))/grid
        print('shufts',shifts)
        x_truncated_spline = x_spline - shifts
    
        # truncate left at zero
        num_negatives = sum(x_truncated_spline < 0)
        x_truncated_spline = x_truncated_spline[num_negatives:]
        truncated_spline[num_negatives] += np.sum(truncated_spline[:num_negatives])/grid
        truncated_spline = truncated_spline[num_negatives:]
   
        # normalize again        
        truncated_spline = truncated_spline /np.sum(truncated_spline)*grid

        x_spline = x_truncated_spline
        binomial_spline = truncated_spline
        max_degrees[j] = k_star

    if plot:
        plt.legend()

    return max_degrees

################################################################################

num_trials = 100
remove_nodes = 'attack'

################################################################################

labels = [r'$c_{sim}$', r'$c_{epATA}$', r'$c_{sim+epATA}$', r'$K_{sim}$', r'$K_{from current}$', r'$K_{iteration}$', r'$K_{from first}$']
colors = ['navy', 'mediumblue', 'lightblue', 'red', 'tomato', 'darkorange', 'orange']

################################################################################

prange = [0.9]

for n in [10]:
    data_array = np.zeros((num_trials, 8, n), dtype=float)
    data_array[0] = np.arange(n)

    for ip, p in enumerate(prange):
        
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
                    print('n,p', g.number_of_nodes(), p_new)                    
                    #print('(n, p_new, emd) = ({},{},{})'.format(n, p_new, data_array[j, 6, i]))

                    # calculate max degree with independence assumption but no ER assumption: ("from first")
                    # we calculate the EMD from the Nth largest degree in original network
                    if i <= n:
                        data_array[j, 7, i] = expectedNthLargestDegree(n,  p, i+1)# - i*p
                        data_array[j, 7, i] = data_array[j, 7, i] - np.sum([data_array[j, 7, i_]/(n-i_-1) for i_ in range(i)])
                        if data_array[j, 7, i] < 0:
                            data_array[j, 7, i] = 0

                        #pairs = [[a/(n-1-ia),1-a/(n-1-ia)] 
                        #         for ia, a in enumerate(data_array[j, 7, :i-1])]
                        #pair_values = [[1,0] for _ in data_array[j, 7, :i-1]]
                        #penalty = 0
                        #for p2, pv in zip(itertools.product(*pairs), itertools.product(*pair_values)):
                        #    penalty += np.product(p2)*np.sum(pv)
                        #
                        #data_array[j, 7, i] = data_array[j, 7, i] - penalty


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

        # discard all botched configurations
        for j in range(num_trials):
            for i in range(n):
                if np.isnan(np.mean(data_array[j,:,i])):
                    data_array[j,:,i] += np.nan

        plt.subplot(1,len(prange),1+ip)
        plt.title('p={}'.format(p))
        for l in range(4,len(data_array[0])): #range(1,len(data_array[0])):
            if l not in [90]:
                plt.plot(
                    #np.cumsum(
                    np.nanmean(data_array[:,l],axis=0), #), 
                    marker='ovsd'[l%4], fillstyle='none',
                    ls=['--','-'][l//4], lw=0, c=colors[l-1], label=labels[l-1])
            #print(np.nanmean(data_array, axis=0))

    plt.plot(list(geometric_approach(n, p, n-1))+[0], label='geometric', marker='x')
    plt.legend()
    plt.subplots_adjust(top=0.8)
    plt.savefig('test_average_degree2_n{}.pdf'.format(n))
    plt.clf()
