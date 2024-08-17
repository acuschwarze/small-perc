###############################################################################
#
# Library of helper functions for the robustness library.
#
# This library contains the following functions:
#    degreeProbability (previously "degree_distr")
#    expectedNodeNumber
#    expectedMaxDegree (previously "maxdegree")
#    edgeProbabilityAfterTargetedAttack
#    sampleNetwork (previously "construct_a_network")
#    LaplacianMatrix (previously "laplacian_matrix")
#    getLCC
#
###############################################################################

import os
import networkx as nx
import numpy as np
from scipy.stats import binom as binomialDistribution
import scipy.special
import math

def string2array(s, sep=" "):
    list_of_nums = [float(x) for x in s.strip('[] ').split(sep) if x != '']
    return np.array(list_of_nums)


def degreeFraction(k, G):
    '''Return the value of the fraction of nodes in the graph G that have 
    degree k.
    
    Parameters
    ----------
    k : int
       A node degree.
    
    G : a networkX graph
       graph in which fraction of nodes with degree k is evaluated.
       
    Returns
    -------
    fraction (float)
       The value of the fraction of nodes in G that have degree k.
    '''

    # get the graph's degree sequence
    degree_sequence = [d for n, d in G.degree()]

    # count nodes with of degree k
    degree_count = degree_sequence.count(k)

    # get fraction of nodes that have degree k
    fraction = degree_count / G.number_of_nodes()

    return fraction
    

def expectedNodeNumber(n, p, k):
    '''Expected value of the number of nodes with degree k in an Erdos--Renyi
    graph with n nodes and edge probability p.

    Parameters
    ----------
    n : int
       Number of nodes.
    
    p : float
       Edge probability in Erdos Renyi graph.
       
    k : int
       A node degree.
       
    Returns
    -------
    expected_number (float)
       The expected number of nodes with degree k (does not need to be an
       integer).
    '''
    degree_probability = binomialDistribution(n, p).pmf(k)
    expected_number = n * degree_probability
    
    return expected_number

#    correct one
def expectedMaxDegree(n, p):
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
    probs_at_least_one_node = 1 - (1 - probs_at_least_k) ** (n)

    # every node has at least degree zero
    #probs_at_least_one_node[0] = 1
    # at least one node has degree 1 if the graph is not empty
    #probs_at_least_one_node[1] = 1 - binomialDistribution.pmf(0, n * (n - 1) / 2, p)

    probs_at_least_one_node = np.concatenate([probs_at_least_one_node, [0]])
    probs_kmax = probs_at_least_one_node[:-1] - probs_at_least_one_node[1:]
    mean_k_max = np.sum([probs_kmax[k] * k for k in range(n)])

    return mean_k_max


# def probs_less(n,p,k):
#     if n == 1:
#         prob = 1
#     else:
#         k_possibilities = 0
#         for i in range(k):
#             k_possibilities += scipy.special.comb(n-1,i)
#         prob = probs_less(n-1,p,k)*(p**(n-1))*((1-p)**(0))*k_possibilities
#     return prob

# def probs_less(n,p,k):
#     if k == 0:
#         prob = 0
#     if k == n-1:
#         prob = 1
#     if n == 1:
#         prob = 1
#     else:
#         k_possibilities = 0
#         for i in range(k):
#             k_possibilities += scipy.special.comb(n-1,i) * (1-p)**(n-1-i) * p**(i) * probs_less(n-2,p,k-1)
#         #  (binomialDistribution.cdf(k-1, n - 1, p))**i
#         prob = probs_less(n-1,p,k)
    
#     return prob


# def pi(n,p,j):
#     sum = 0
#     for i in range(n-1):
#         sum += i/(n-1) * binomialDistribution(n)

# def probs_less(n,p,k): # probability that in a G(n,p), all nodes have degree < k
#     print("n p k",n,p,k)
#     if k == 0:
#         prob = 0
#     if k == n-1:
#         prob = 1
#     if n == 1:
#         prob = 1
#     else:
#         k_possibilities = 0
#         for i in range(k):
#             for j in range(n): # j for all values of exactly how many nodes in n-1 have less than k-1 degree
#                 x = 0
#                 for i_x in range(j):
#                     #x += 1-((1-binomialDistribution.cdf(i_x,n-1,p))**(i_x)*scipy.special.comb(n-1,i_x)) # probability that there are at least j nodes with less than k-1 
#                     x += binomialDistribution.cdf(k-2,n-1,p) ** (i_x) * scipy.special.comb(n-1,i_x)
#                 k_possibilities += probs_less(n-1,p,i) * scipy.special.comb(j,i) * p**i * (1-p)**(n-1-i) * (x)
#             #k_possibilities += sum / i
#         #  (binomialDistribution.cdf(k-1, n - 1, p))**i
#         prob = k_possibilities
    
#     return prob

# def expectedMaxDegree(n,p):
#     if n in [0, 1] or p == 0:
#         return 0
    
#     array = np.zeros(n)
#     for j in range(n):
#         array[j] = 1-probs_less(n,p,j)
#     array = np.concatenate([array, [0]])
#     print("array",array)
#     probs_kmax = array[:-1] - array[1:]
#     mean_k_max = np.sum([probs_kmax[k]*k for k in range(n)])
#     print("probs_kmax",probs_kmax)
#     print("meankmax",mean_k_max)
#     return mean_k_max


# def expectedMaxDegree(n, p):
#     '''Calculate expected value of the maximum degree in an Erdos--Renyi graph
#     with n nodes and edge probability p.

#     Parameters
#     ----------
#     n : int
#        Number of nodes.

#     p : float
#        Edge probability in Erdos Renyi graph.

#     Returns
#     -------
#     emd (int)
#        The expected value of the maximum degree.
#     '''

#     probs_k_or_less = np.array([binomialDistribution.cdf(k,n-1,p) for k in range(n)])
#     probs_at_least_k = np.concatenate([[1],np.array(1-probs_k_or_less[:-1])])
#     probs_at_least_one_node = 1-(1-probs_at_least_k)**(1)
#     probs_at_least_one_node = np.concatenate([probs_at_least_one_node, [0]])
#     probs_kmax = probs_at_least_one_node[:-1]-probs_at_least_one_node[1:]
#     mean_k_max = np.sum([probs_kmax[k]*k for k in range(n)])
#     print(n,p,mean_k_max)
#     return mean_k_max

    

def edgeProbabilityAfterTargetedAttack(n, p):
    '''Calculate edge probability in an Erdos--Renyi network with original size
    `n` and original edge probability `p` after removing the node with the
    highest degree.

    Parameters
    ----------
    n : int
       Number of nodes.
    
    p : float
       Edge probability in Erdos Renyi graph.
       
    Returns
    -------
    new_p (float)
       Updated edge probability.
    '''
    if n <=2:
        new_p = 0

    else:
        emd = expectedMaxDegree(n, p)
        new_p = p * n / (n - 2) - 2 * emd / ((n - 1) * (n - 2))
        new_p = max([new_p, 0])

    return new_p


def sampleNetwork(n, p, graph_type='ER'):
    '''Sample a network with `n` nodes and `m` edges per node from the Erdos--
    Renyi model or the Barabasi--Albert model.

    ISSUE #1: NetworkX's BA algorithm is flaky with the number of edges!
    
    Parameters
    ----------
    n : int
       Number of nodes.
    
    p : float
       Edge probability.
       
    graph_type : str
       If graph_type=='ER', return an Erdos--Renyi graph; if graph_type=='BA',
       return a Barabasi--Albert graph.
       
    Returns
    -------
    g : a networkX graph
       An undirected graph with n nodes and m*n(?) edges.
    '''

    if graph_type == 'ER':
        # generate an Erdos--Renyi random graph
        g = nx.erdos_renyi_graph(n, p, seed=None, directed=False)
        return g

    elif graph_type == 'SF':
        # generate a scale-free network using the Barabasi--Albert model
        # since no initial graph is given, the algorithm starts with a 
        # star graph with m+1 nodes
        g = nx.barabasi_albert_graph(n, int(np.round(p*(n-1))))
        return g
    else:
        raise ValueError("Invalid graph_type")


def LaplacianMatrix(G):
    '''Construct the combinatorial Laplacian matrix for a graph `G`.

    Parameters
    ----------
    G : a networkX graph
       A graph.

    Returns
    -------
    L : 2D numpy array
       The Laplacian matrix of the graph G.
    '''
    L = nx.laplacian_matrix(G).toarray()

    return L


def getLCC(G):
    '''Get the largest connected component of a graph `G`.

    Parameters
    ----------
    G : a networkX graph
       A graph.

    Returns
    -------
    g : a networkX graph
       The largest connected component of a graph G.
    '''

    node_sets = sorted(nx.connected_components(G), key=len)
    lcc_set = node_sets[-1]
    g = G.subgraph(lcc_set).copy()

    return g


def relSCurve_precalculated(n, p, targeted_removal=False, simulated=False, finite=True):
    """
    Retrieve the finite percolation data from precalculated files for network 
    sizes 1 to 100 and probabilities between 0.01 and 1.00 (in steps of 0.01).

    If `simulated` is `False`, this function retrieves k-th row of data from
    the 2D numpy array stored in the file 
    "data/synthetic_data/relSCurve_attack{targeted_removal}_n{n}.npy"
    where k is the closest integer to p/0.01.

    If `simulated` is `True`, this function retrieves k-th slice of data from
    the 3D numpy array stored in the file 
    "data/synthetic_data/simRelSCurve_attack{targeted_removal}_n{n}.npy"
    where k is the closest integer to p/0.01.

    Parameters:
    - n (int): The number of nodes.
    - p (float): The probability value.
    - targeted_removal (bool, optional): Whether the removal is targeted. 
        Default is False.
    - simulated (bool, optional): Whether to retrieve simulated data. 
        Default is False.

    Returns:
    - numpy.ndarray: 1D of length n+1 or 2D array of shape (n+1,100)
    """

    # Define the path to the data file
    if simulated:
        fstring = "simRelSCurve"
    elif finite:
        fstring = "relSCurve"
    else:
        fstring = "infRelSCurve"

    file_name = "{}_attack{}_n{}.npy".format(fstring, targeted_removal, n)

      
    file_path = os.path.join("data", "synthetic_data", file_name)

    # Load the numpy array from the file
    data_array = np.load(file_path)

    # Calculate the row index k
    k = int(round(p / 0.01))-1

    # Retrieve the k-th row from the data array
    if k < 0 or k >= data_array.shape[0]:
        shape = data_array.shape
        verr = "p={} is out of bounds for array with shape {}".format(p,shape)
        raise ValueError(verr)

    return data_array[k]