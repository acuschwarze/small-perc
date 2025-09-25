###############################################################################
#
# Network Robustness Analysis Library
#
# Functions:
#    execute_subprocess - Execute an external program and return its output
#    load_recursion_cache - Load cached data for finite-theory calculations
#    string_to_array - Convert string representation to numpy array
#    degree_fraction - Calculate fraction of nodes with specific degree
#    expected_node_count - Expected nodes with degree k in Erdos-Renyi graph
#    expected_max_degree - Expected maximum degree in Erdos-Renyi graph
#    edge_probability_after_attack - Edge probability after targeted removal
#    sample_network - Generate random network (Erdos-Renyi or Barabasi-Albert)
#    laplacian_matrix - Compute graph Laplacian matrix
#    get_largest_component - Extract largest connected component
#    load_percolation_curve - Load precalculated percolation data
#
###############################################################################

# Import libraries
import os, pickle
import networkx as nx
import numpy as np
from scipy.stats import binom as binomial_dist
from typing import List, Optional, Tuple
import subprocess
from pathlib import Path

# Add the parent directory to the path to import local libraries
REPO_ROOT = str(Path(__file__).parent.parent)
SYNTH_PATH = os.path.join(REPO_ROOT, 'data-synthetic')


def execute_subprocess(executable_path: List[str], return_error=False) -> Optional[str]:
    """Execute an external program and return its output.
    
    Parameters
    ----------
    executable_path : List[str]
        Path to executable and its arguments
        
    Returns
    -------
    Optional[str]
        Output from the executable or None if error
    """
    try:
        result = subprocess.run(executable_path, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        result = subprocess.run(executable_path, capture_output=True, text=True, check=True)
        if return_error:
            return result.stdout.strip()
        else:
            return None
    

def load_recursion_cache(path: str, connectivity_cache_name: str='fvalues.p', 
   probability_cache_name: str='Pvalues.p') -> Tuple:
   """ load connectivity cache and probability cache for finite-theory calculations
   """

   output = (pickle.load(open(os.path.join(path, connectivity_cache_name), 'rb')),
        pickle.load(open(os.path.join(path, probability_cache_name), 'rb')))
   return output


def string_to_array(text: str, separator: str = " ") -> np.ndarray:
    """Convert string representation to numpy array."""
    values = [float(x) for x in text.strip('[] ').split(separator) if x != '']
    return np.array(values)


def degree_fraction(degree: int, graph: nx.Graph) -> float:
    """Return fraction of nodes with specified degree.
    
    Parameters
    ----------
    degree : int
       Target node degree
    graph : nx.Graph
       Graph to analyze
       
    Returns
    -------
    float
       Fraction of nodes with specified degree
    """
    degree_sequence = [d for _, d in graph.degree()] 
    count = degree_sequence.count(degree)
    return count / graph.number_of_nodes()
    

def expected_node_count(nodes: int, edge_prob: float, degree: int) -> float:
    """Expected number of nodes with degree k in Erdos-Renyi graph.

    Parameters
    ----------
    nodes : int
       Number of nodes
    edge_prob : float
       Edge probability
    degree : int
       Target degree
       
    Returns
    -------
    float
       Expected number of nodes with specified degree
    """
    prob = binomial_dist(nodes, edge_prob).pmf(degree) 
    return nodes * prob


def expected_max_degree(nodes: int, edge_prob: float) -> float:
    """Calculate expected maximum degree in Erdos-Renyi graph.

    Parameters
    ----------
    nodes : int
       Number of nodes
    edge_prob : float
       Edge probability

    Returns
    -------
    float
       Expected maximum degree
    """
    if nodes in [0, 1] or edge_prob == 0:
        return 0

    if nodes == 2:
        return edge_prob
        
    prob_at_least_k = np.cumsum([binomial_dist.pmf(k, nodes - 1, edge_prob) 
                                  for k in range(nodes)][::-1])[::-1]
    prob_at_least_one = 1 - (1 - prob_at_least_k) ** nodes
    prob_at_least_one = np.concatenate([prob_at_least_one, [0]])
    prob_max = prob_at_least_one[:-1] - prob_at_least_one[1:]
    mean_max = np.sum([prob_max[k] * k for k in range(nodes)])

    return mean_max


def edge_probability_after_attack(nodes: int, edge_prob: float) -> float:
    """Edge probability after removing highest-degree node.

    Parameters
    ----------
    nodes : int
       Original number of nodes
    edge_prob : float
       Original edge probability
       
    Returns
    -------
    float
       Updated edge probability
    """
    if nodes <= 2:
        return 0

    max_deg = expected_max_degree(nodes, edge_prob)
    new_prob = edge_prob * nodes / (nodes - 2) - 2 * max_deg / ((nodes - 1) * (nodes - 2))
    return max(new_prob, 0)


def sample_network(nodes: int, prob: float, graph_type: str = 'ER') -> nx.Graph:
    """Generate random network.

    Parameters
    ----------
    nodes : int
       Number of nodes
    prob : float
       Edge probability (ER) or attachment parameter (BA)
    graph_type : str
       'ER' for Erdos-Renyi, 'SF' for scale-free Barabasi-Albert
       
    Returns
    -------
    nx.Graph
       Generated network
    """
    if graph_type == 'ER':
        return nx.erdos_renyi_graph(nodes, prob, directed=False)
    elif graph_type == 'SF':
        m = int(np.round(prob * (nodes - 1)))
        return nx.barabasi_albert_graph(nodes, m)
    else:
        raise ValueError("Invalid graph_type")


def laplacian_matrix(graph: nx.Graph) -> np.ndarray:
    """Compute combinatorial Laplacian matrix.

    Parameters
    ----------
    graph : nx.Graph
       Input graph

    Returns
    -------
    np.ndarray
       Laplacian matrix
    """
    return nx.laplacian_matrix(graph).toarray()


def get_largest_component(graph: nx.Graph) -> nx.Graph:
    """Extract largest connected component.

    Parameters
    ----------
    graph : nx.Graph
       Input graph

    Returns
    -------
    nx.Graph
       Largest connected component
    """
    components = sorted(nx.connected_components(graph), key=len)
    largest = components[-1]
    return graph.subgraph(largest).copy()


def load_percolation_curve(nodes: int, prob: float, targeted_removal: bool = False, 
                           simulated: bool = False, finite: bool = True) -> np.ndarray:
    """Load precalculated percolation data.

    Retrieves percolation data for networks with 1-100 nodes and 
    probabilities 0.01-1.00 (in 0.01 steps).

    Parameters
    ----------
    nodes : int
       Number of nodes
    prob : float
       Probability value
    targeted : bool
       Whether removal is targeted
    simulated : bool
       Whether to retrieve simulated data
    finite : bool
       Whether to use finite percolation data

    Returns
    -------
    np.ndarray
       1D array of length nodes+1 or 2D array of shape (nodes+1, 100)
    """
    if simulated:
        prefix = "simRelSCurve"
    elif finite:
        prefix = "relSCurve"
    else:
        prefix = "infRelSCurve"

    filename = f"{prefix}_attack{targeted_removal}_n{nodes}.npy"
    filepath = os.path.join(SYNTH_PATH, filename)
    
    data = np.load(filepath)
    index = int(round(prob / 0.01)) - 1

    if index < 0 or index >= data.shape[0]:
        raise ValueError(f"p={prob} is out of bounds for array with shape {data.shape}")

    return data[index]