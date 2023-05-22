###############################################################################
#
# Library of helper functions for the robustness library.
#
# This library contains the following functions:
#    degreeProbability (previously "degree_distr")
#    expectedNodeNumber
#    expectedMaxDegree (previously "maxdegree")
#    sampleNetwork (previously "construct_a_network")
#    LaplacianMatrix (previously "laplacian_matrix")
#    getLCC
#
###############################################################################

import networkx as nx
from scipy.stats import binom as binomialDistribution

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
    emd (int)
       The expected value of the maximum degree.
    '''
    
    # get list of possible degrees and expected number of nodes
    vals = [[k, expectedNodeNumber(n, p, k)] for k in range(1, n + 1)]

    # return largest value of k for which the expected number of nodes is
    # greater than or equal to 1
    for k, enn in vals[::-1]:
        if enn >= 1:
            return k
    # otherwise return 0
    return 0
    

def sampleNetwork(n, m, graph_type='ER'):
    '''Sample a network with `n` nodes and `m` edges per node from the Erdos--
    Renyi model or the Barabasi--Albert model.
    
    ISSUE #1: For ER graphs the edge probability is currently constant!
    ISSUE #2: NetworkX's BA algorithm is flaky with the number of edges!
    
    Parameters
    ----------
    n : int
       Number of nodes.
    
    m : int or float
       (Expected) number of edges per node.
       
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
        p = .1 # AS: This does not seem right!
        g = nx.erdos_renyi_graph(n, p, seed=None, directed=False)
        return g

    elif graph_type == 'SF':
        # generate a scale-free network using the Barabasi--Albert model
        # since no initial graph is given, the algorithm starts with a 
        # star graph with m+1 nodes
        g = nx.barabasi_albert_graph(n, m)
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