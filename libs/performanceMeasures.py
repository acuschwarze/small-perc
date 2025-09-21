###############################################################################
#
# Library of functions to compute graph performance measures.
#
# Functions:
#     averageDegree: Compute mean degree of a graph
#     getEfficiency: Calculate graph efficiency (inverse path length average)
#     meanShortestPathLength: Compute mean shortest path length
#     meanCommunicability: Calculate natural connectivity via matrix exponential
#     resistanceDistance: Compute resistance distance using Laplacian pseudo-inverse
#     getReachability: Calculate fraction of connected node pairs
#     sizeOfLCC: Get size of largest connected component
#     relativeSizeOfLCC: Get relative size of largest connected component
#     getEntropy: Calculate entropy of degree distribution
#     averageComponentSize: Compute mean component size
#     averageSmallComponentSize: Compute mean size of non-LCC components
#
###############################################################################

import numpy as np
import networkx as nx
from scipy.special import comb
from itertools import combinations
from utils import *

class PerformanceGraph(nx.Graph):

    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, **attr)
        self.performance_metrics = {}

    def getAverageDegree(self, recompute : bool = False) -> float:
        '''Get average degree of a graph.

        Parameters
        ----------
        G : networkx.Graph
            A graph.

        Returns
        -------
        float
            Mean degree of the graph.
        '''
        if recompute or "average_degree" not in self.performance_metrics.keys():
            self.computeAverageDegree()

        return self.performance_metrics["average_degree"]
    
    def computeAverageDegree(self):
        average_degree = self.number_of_edges() * 2 / self.number_of_nodes()
        self.performance_metrics["average_degree"]
        return average_degree


def getEfficiency(G: nx.Graph, lcc_only: bool = False) -> float:
    '''Get efficiency of a graph.

    Parameters
    ----------
    G : networkx.Graph
        A graph.
    lcc_only : bool, default=False
        If True, compute efficiency only on the largest connected component.

    Returns
    -------
    float
        Efficiency of the graph.
    '''
    if lcc_only:
        return getEfficiency(getLCC(G), lcc_only=False)

    n = G.number_of_nodes()
    if n < 2:
        return 0

    lengths = dict(nx.all_pairs_shortest_path_length(G))
    
    sum_efficiencies = 0
    for i, j in combinations(G.nodes(), 2):
        if j in lengths[i].keys():
            sum_efficiencies += 1 / lengths[i][j]

    return sum_efficiencies / (n * (n - 1))


def meanShortestPathLength(G: nx.Graph, lcc_only: bool = True) -> float:
    '''Get the mean shortest path length of a graph.

    Parameters
    ----------
    G : networkx.Graph
        A graph.
    lcc_only : bool, default=True
        If True, compute only on the largest connected component.

    Returns
    -------
    float
        Mean shortest path length of the graph.
    '''
    if lcc_only:
        return nx.average_shortest_path_length(getLCC(G))
    else:
        raise NotImplementedError(
            'No mean-shortest-path-length computation implemented for'
            ' fragmented networks.')


def meanCommunicability(G: nx.Graph, lcc_only: bool = False) -> float:
    '''Get the mean communicability (natural connectivity) of a graph.

    Parameters
    ----------
    G : networkx.Graph
        A graph.
    lcc_only : bool, default=False
        If True, compute only on the largest connected component.

    Returns
    -------
    float
        Mean communicability of the graph.
    '''
    if lcc_only:
        return meanCommunicability(getLCC(G))

    n = G.number_of_nodes()
    if n < 2:
        return 0

    adjacency = nx.to_numpy_array(G)
    exp_adjacency = np.linalg.expm(adjacency)
    return np.log(np.trace(exp_adjacency)) - np.log(n)


def resistanceDistance(G: nx.Graph, lcc_only: bool = False) -> float:
    '''Get the resistance distance of a graph.

    Parameters
    ----------
    G : networkx.Graph
        A graph.
    lcc_only : bool, default=False
        If True, compute only on the largest connected component.

    Returns
    -------
    float
        Resistance distance of the graph.
    '''
    if lcc_only:
        return resistanceDistance(getLCC(G))

    n = G.number_of_nodes()
    if n < 1:
        return 0

    laplacian = LaplacianMatrix(G)
    laplacian_pinv = np.linalg.pinv(laplacian)
    return n * np.trace(laplacian_pinv)


def getReachability(G: nx.Graph) -> float:
    '''Get the reachability of a graph.

    Parameters
    ----------
    G : networkx.Graph
        A graph.

    Returns
    -------
    float
        Reachability (fraction of connected node pairs).
    '''
    n = G.number_of_nodes()
    if n == 0:
        return 0
        
    connected_pairs = 0
    for i, j in combinations(G.nodes(), 2):
        if nx.has_path(G, i, j):
            connected_pairs += 1
    
    return connected_pairs / (2 * comb(n, 2))


def sizeOfLCC(G: nx.Graph) -> int:
    '''Get the size of the largest connected component.

    Parameters
    ----------
    G : networkx.Graph
        A graph.

    Returns
    -------
    int
        Size of the largest connected component.
    '''
    if G.number_of_nodes() == 0:
        return 0
    return len(max(nx.connected_components(G), key=len))


def relativeSizeOfLCC(G: nx.Graph) -> float:
    '''Get the relative size of the largest connected component.

    Parameters
    ----------
    G : networkx.Graph
        A graph.

    Returns
    -------
    float
        Relative size of the largest connected component.
    '''
    n = G.number_of_nodes()
    if n == 0:
        return 0
    return sizeOfLCC(G) / n


def getEntropy(G: nx.Graph) -> float:
    '''Get the entropy of the degree distribution.

    Parameters
    ----------
    G : networkx.Graph
        A graph.

    Returns
    -------
    float
        Entropy of the degree distribution.
    '''
    if nx.number_of_nodes(G) == 0:
        return 0
        
    max_degree = sorted(G.degree, key=lambda x: x[1], reverse=True)[0][1]
    entropy = 0
    
    for k in range(max_degree):
        pk = degreeFraction(k, G)
        if pk > 0:
            entropy += -pk * np.log(pk)
    
    return entropy


def averageComponentSize(G: nx.Graph) -> float:
    '''Get the average component size of a graph.

    Parameters
    ----------
    G : networkx.Graph
        A graph.

    Returns
    -------
    float
        Average component size.
    '''
    n_components = nx.number_connected_components(G)
    if n_components == 0:
        return 0
    return nx.number_of_nodes(G) / n_components


def averageSmallComponentSize(G: nx.Graph) -> float:
    '''Get the average size of non-LCC components.

    Parameters
    ----------
    G : networkx.Graph
        A graph.

    Returns
    -------
    float
        Average size of components excluding the largest.
    '''
    n = nx.number_of_nodes(G)
    lcc_size = sizeOfLCC(G)
    nodes_not_in_lcc = n - lcc_size
    n_small_components = nx.number_connected_components(G) - 1
    
    if n_small_components == 0:
        return 0
    return nodes_not_in_lcc / n_small_components