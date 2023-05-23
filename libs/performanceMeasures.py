###############################################################################
#
# Library of functions to compute performance measures.
#
# This library contains the following functions:
#     averageDegree (previously "average_degree")
#     getEfficiency (previously "compute_efficiency")
#     meanShortestPathLength(previously "mean_shortest_path")
#     meanCommunicability (previously "communicability")
#     resistanceDistance
#     getReachability
#     sizeOfLCC
#     relativeSizeOfLCC
#     getEntropy
#     averageComponentSize
#     averageSmallComponentSize
#
###############################################################################

import numpy as np
import networkx as nx
from scipy.special import comb
from itertools import combinations
#from data import *
from utils import *

def averageDegree(G):
    '''Get average degree of a graph `G`.

    Parameters
    ----------
    G : a networkX graph
       A graph.

    Returns
    -------
    md : float
       Mean degree of the graph G.
    '''
    md = G.number_of_edges() * 2 / G.number_of_nodes()

    return md


def getEfficiency(G, lcc_only=False):
    '''Get efficiency of a graph `G`.
    
    ISSUE #1: Why do we set the efficiency for small graphs to 0 instead of 1?

    Parameters
    ----------
    G : a networkX graph
       A graph.
    
    lcc_only : bool (default=False)
       If lcc_only is True, compute efficiency only on the largest connected
       component of G.

    Returns
    -------
    effi : float
       Efficiency of the graph G.
    '''
    if lcc_only:
        return getEfficiency(getLCC(G), lcc_only=False)

    # get number of nodes
    n = G.number_of_nodes()

    # set special case for small graphs (AS: is this necessary?)
    if n < 2:
        return 0

    # get shortest path lengths
    lengths = dict(nx.all_pairs_shortest_path_length(G))

    # count and sum pairwise efficiencies
    sum_efficiencies = 0
    for i,j in combinations(G.nodes(),2):
        if j in lengths[i].keys():
            sum_efficiencies += 1/lengths[i][j]

    # get mean of pairwise efficiencies
    effi = sum_efficiencies / (n*(n-1))

    return effi


def meanShortestPathLength(G, lcc_only=True):
    '''Get the mean-shortest-path length of a graph `G`.

    Parameters
    ----------
    G : a networkX graph
       A graph.
    
    lcc_only : bool (default=True)
       If lcc_only is True, compute the mean-shortest-path length only on the 
       largest connected component of G.

    Returns
    -------
    mspl : float
       Mean-shortest-path length of the graph G.
    '''

    if lcc_only:
        mspl = nx.average_shortest_path_length(getLCC(G))
        return mspl

    else:
        raise NotImplementedError(
            'No mean-shortest-path-length computation implemented for'
            +' fragmented networks.')


def meanCommunicability(G, lcc_only=False):
    '''Get the mean communicability (i.e., the natural connectivity) of a graph
    `G`.

    Parameters
    ----------
    G : a networkX graph
       A graph.
    
    lcc_only : bool (default=False)
       If lcc_only is True, compute the mean communicability only on the 
       largest connected component of G.

    Returns
    -------
    comm : float
       Mean communicability of the graph G.
    '''
    
    if lcc_only:
        return meanCommunicability(getLCC(G))

    n = G.number_of_nodes()
    if n < 2:
        return 0

    # get adjacency matrix
    A = nx.to_numpy_array(G)
    # get matrix exponential of the adjacency matrix
    expA = np.linalg.expm(A)
    # get communicability
    comm = np.log(np.trace(expA)) - np.log(n)

    return comm


def resistanceDistance(G,lcc_only=False):
    '''Get the resistance distance of a graph `G`.

    Parameters
    ----------
    G : a networkX graph
       A graph.

    lcc_only : bool (default=False)
       If lcc_only is True, compute the resistance distance only on the
       largest connected component of G.

    Returns
    -------
    rd : float
       Resistance distance of the graph G.
    '''

    if lcc_only:
        return resistanceDistance(getLCC(G))

    # get number of nodes
    n = G.number_of_nodes()
    
    # set special case for small networks
    if n < 1:
        return 0

    # get pseudo-inverse of the Laplacian matrix
    L = LaplacianMatrix(G)
    L_plus = np.linalg.pinv(L)

    # get resistance distance
    rd =  n * np.trace(L_plus)

    return rd


def getReachability(G):
    '''Get the reachability of a graph `G`.

    Parameters
    ----------
    G : a networkX graph
       A graph.

    Returns
    -------
    r : float
       Reachability of the graph G.
    '''
    # get number of nodes
    n = G.number_of_nodes()

    # count number of connected node pairs
    r = 0
    if n > 0:
        for i, j in combinations(G.nodes(), 2):
            if nx.has_path(G, i, j):
                r += 1
        # divide by number of node pairs
        r = r / (2*comb(n, 2))

    return r
    

def sizeOfLCC(G):
    '''Get the size of the largest connected component of a graph `G`.

    Parameters
    ----------
    G : a networkX graph
       A graph.

    Returns
    -------
    lcc_size : float
       Size of the largest connected component of the graph G.
    '''
    if (G.number_of_nodes() == 0):
        return 0
    lcc_size = len(max(nx.connected_components(G), key=len))

    return lcc_size


def relativeSizeOfLCC(G):
    '''Get the size of the relative largest connected component of a graph 
    `G`.

    Parameters
    ----------
    G : a networkX graph
       A graph.

    Returns
    -------
    rel_size : float
       Relative size of the largest connected component of the graph G.
    '''
    # get number of nodes
    n = G.number_of_nodes()
    # get size of the LCC
    rel_size = sizeOfLCC(G)
    if n > 0:
        # divide LCC size by number of nodes if G is not empty
        rel_size = rel_size / n

    return rel_size
    

def getEntropy(G):
    '''Get the entropy (of the degree distribution) of a graph `G`.

    Parameters
    ----------
    G : a networkX graph
       A graph.

    Returns
    -------
    H : float
       Entropy (of the degree distribution) of the graph G.
    '''
    if nx.number_of_nodes(G) == 0:
        return 0
    max_degree = sorted(G.degree, key=lambda x: x[1], reverse=True)[0][1]
    H = 0
    for k in range(max_degree):
        pk = degreeFraction(k, G)
    if pk > 0:
        H += -pk * np.log(pk)

    return H
    

def averageComponentSize(G):
    '''Get the average (i.e., mean) component size of a graph `G`.

    Parameters
    ----------
    G : a networkX graph
       A graph.

    Returns
    -------
    acs : float
       Average (i.e., mean) component size of a graph `G`.
    '''

    n = nx.number_of_nodes(G)
    n_c = nx.number_connected_components(G)
    if n_c == 0:
        return 0
    acs =  n / n_c
    return acs


def averageSmallComponentSize(G):
    '''Get the average (i.e., mean) component size of a graph `G`.

    Parameters
    ----------
    G : a networkX graph
       A graph.

    Returns
    -------
    ascs : float
       Average (i.e., mean) component size of a graph `G`.
    '''
    n = nx.number_of_nodes(G)
    lcc_size = sizeOfLCC(G)
    n_no_lcc = n - lcc_size
    n_c = nx.number_connected_components(G) - 1
    if n_c == 0:
        return 0
    ascs = n_no_lcc / n_c

    return ascs

