###############################################################################
#
# Library of functions to compute network performance measures.
#
# Functions included:
#     average_degree           - Compute mean degree of nodes in graph
#     efficiency              - Compute network efficiency (inverse path lengths)
#     mean_shortest_path_length - Compute average shortest path length
#     mean_communicability    - Compute mean communicability (natural connectivity)
#     resistance_distance     - Compute total effective resistance
#     reachability           - Compute fraction of connected node pairs
#     size_of_lcc            - Get size of largest connected component
#     relative_size_of_lcc   - Get relative size of largest connected component
#     entropy                - Compute entropy of degree distribution
#     average_component_size  - Compute mean size of all components
#     average_small_component_size - Compute mean size of non-LCC components
#
###############################################################################

# Import libraries
import sys
from pathlib import Path
import numpy as np
import networkx as nx
from scipy.special import comb
from itertools import combinations

# Add the parent directory to the path to import local libraries
REPO_ROOT = str(Path(__file__).parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Import from local libraries
from libs.utils import laplacian_matrix, degree_fraction, get_largest_component


def average_degree(graph: nx.Graph) -> float:
    """Get average degree of a graph.

    Parameters
    ----------
    graph : nx.Graph
        A networkX graph.

    Returns
    -------
    float
        Mean degree of the graph.
    """
    return graph.number_of_edges() * 2 / graph.number_of_nodes()


def efficiency(graph: nx.Graph, lcc_only: bool = False) -> float:
    """Get efficiency of a graph (mean inverse shortest path length).

    Parameters
    ----------
    graph : nx.Graph
        A networkX graph.
    lcc_only : bool, optional
        If True, compute efficiency only on largest connected component.

    Returns
    -------
    float
        Efficiency of the graph.
    """
    if lcc_only:
        return efficiency(get_largest_component(graph), lcc_only=False)

    n_nodes = graph.number_of_nodes()
    if n_nodes < 2:
        return 0

    lengths = dict(nx.all_pairs_shortest_path_length(graph))
    
    sum_efficiencies = 0
    for i, j in combinations(graph.nodes(), 2):
        if j in lengths[i]:
            sum_efficiencies += 1 / lengths[i][j]

    return sum_efficiencies / (n_nodes * (n_nodes - 1))


def mean_shortest_path_length(graph: nx.Graph, lcc_only: bool = True) -> float:
    """Get mean shortest path length of a graph.

    Parameters
    ----------
    graph : nx.Graph
        A networkX graph.
    lcc_only : bool, optional
        If True, compute only on largest connected component.

    Returns
    -------
    float
        Mean shortest path length.
    """
    if lcc_only:
        return nx.average_shortest_path_length(get_largest_component(graph))
    
    raise NotImplementedError(
        'Mean shortest path length not implemented for fragmented networks.')


def mean_communicability(graph: nx.Graph, lcc_only: bool = False) -> float:
    """Get mean communicability (natural connectivity) of a graph.

    Parameters
    ----------
    graph : nx.Graph
        A networkX graph.
    lcc_only : bool, optional
        If True, compute only on largest connected component.

    Returns
    -------
    float
        Mean communicability of the graph.
    """
    if lcc_only:
        return mean_communicability(get_largest_component(graph))

    n_nodes = graph.number_of_nodes()
    if n_nodes < 2:
        return 0

    adjacency = nx.to_numpy_array(graph)
    exp_adjacency = np.linalg.expm(adjacency) 
    
    return np.log(np.trace(exp_adjacency)) - np.log(n_nodes)


def resistance_distance(graph: nx.Graph, lcc_only: bool = False) -> float:
    """Get resistance distance of a graph.

    Parameters
    ----------
    graph : nx.Graph
        A networkX graph.
    lcc_only : bool, optional
        If True, compute only on largest connected component.

    Returns
    -------
    float
        Resistance distance of the graph.
    """
    if lcc_only:
        return resistance_distance(get_largest_component(graph))

    n_nodes = graph.number_of_nodes()
    if n_nodes < 1:
        return 0

    laplacian = laplacian_matrix(graph)
    laplacian_pinv = np.linalg.pinv(laplacian)
    
    return n_nodes * np.trace(laplacian_pinv)


def reachability(graph: nx.Graph) -> float:
    """Get reachability (fraction of connected pairs) of a graph.

    Parameters
    ----------
    graph : nx.Graph
        A networkX graph.

    Returns
    -------
    float
        Reachability of the graph.
    """
    n_nodes = graph.number_of_nodes()
    if n_nodes == 0:
        return 0
    
    connected_pairs = sum(1 for i, j in combinations(graph.nodes(), 2) 
                          if nx.has_path(graph, i, j))
    result = connected_pairs / (2 * comb(n_nodes, 2)) 
    return float(result)


def size_of_lcc(graph: nx.Graph) -> int:
    """Get size of largest connected component.

    Parameters
    ----------
    graph : nx.Graph
        A networkX graph.

    Returns
    -------
    int
        Size of largest connected component.
    """
    if graph.number_of_nodes() == 0:
        return 0
    
    return len(max(nx.connected_components(graph), key=len))


def relative_size_of_lcc(graph: nx.Graph) -> float:
    """Get relative size of largest connected component.

    Parameters
    ----------
    graph : nx.Graph
        A networkX graph.

    Returns
    -------
    float
        Relative size of largest connected component.
    """
    n_nodes = graph.number_of_nodes()
    if n_nodes == 0:
        return 0
    
    return size_of_lcc(graph) / n_nodes


def entropy(graph: nx.Graph) -> float:
    """Get entropy of degree distribution.

    Parameters
    ----------
    graph : nx.Graph
        A networkX graph.

    Returns
    -------
    float
        Entropy of degree distribution.
    """
    if graph.number_of_nodes() == 0:
        return 0
    
    max_deg = max(dict(graph.degree).values()) 
    entropy_sum = 0
    
    for k in range(max_deg + 1): 
        pk = degree_fraction(k, graph)
        if pk > 0:
            entropy_sum -= pk * np.log(pk)
    
    return entropy_sum


def average_component_size(graph: nx.Graph) -> float:
    """Get average component size of a graph.

    Parameters
    ----------
    graph : nx.Graph
        A networkX graph.

    Returns
    -------
    float
        Average component size.
    """
    n_components = nx.number_connected_components(graph)
    if n_components == 0:
        return 0
    
    return graph.number_of_nodes() / n_components


def average_small_component_size(graph: nx.Graph) -> float:
    """Get average size of components excluding largest.

    Parameters
    ----------
    graph : nx.Graph
        A networkX graph.

    Returns
    -------
    float
        Average size of non-LCC components.
    """
    n_components = nx.number_connected_components(graph) - 1
    if n_components == 0:
        return 0
    
    n_nodes = graph.number_of_nodes()
    lcc_size = size_of_lcc(graph)
    n_non_lcc = n_nodes - lcc_size
    
    return n_non_lcc / n_components