"""
Library for calculating theoretical percolation results for finite networks.

Functions:
    execute_subprocess: Execute an external program and return its output
    connectedness_probability_raw: Calculate probability that a subgraph is connected (no memoization)
    connectedness_probability: Calculate probability that a subgraph is connected (with memoization)
    isolation_probability: Calculate probability that nodes have no external neighbors
    lcc_probability_raw: Calculate probability of largest component size (no memoization)
    lcc_probability: Calculate probability of largest component size (with memoization)
    lcc_probability_external: Calculate probability using external executable
    expected_lcc_size_raw: Calculate expected largest component size (no memoization)
    expected_lcc_size: Calculate expected largest component size (with memoization)
    lcc_sequence: Calculate expected largest component sizes under sequential node removal
    relative_lcc_sequence: Calculate relative largest component sizes under sequential node removal
    relative_lcc_points: Calculate expected largest component sizes for specific network sizes
"""

# Import libraries
import os, sys
import numpy as np
from pathlib import Path
from scipy.special import comb
from typing import Dict, List, Tuple

# Add the parent directory to the path to import local libraries
REPO_ROOT = str(Path(__file__).parent.parent)
CPP_PATH = os.path.join(REPO_ROOT, 'cpp')
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Import from local libraries
from libs.utils import execute_subprocess, edge_probability_after_attack


def connectedness_probability_raw(edge_prob: float, subgraph_size: int, 
                                         network_size: int) -> float:
    """Calculate probability that a subgraph is connected (without memoization).
    
    Parameters
    ----------
    edge_prob : float
        Edge probability in parent graph
    subgraph_size : int
        Number of nodes in subgraph
    network_size : int
        Number of nodes in parent graph
        
    Returns
    -------
    float
        Probability that the subgraph is connected
    """
    if subgraph_size == 1:
        return 1.0
    
    total = 0.0
    for k in range(1, subgraph_size):
        total += (connectedness_probability_raw(edge_prob, k, network_size) * 
                 comb(subgraph_size - 1, k - 1) * 
                 (1 - edge_prob) ** (k * (subgraph_size - k)))
    
    return float(1 - total)


def connectedness_probability(edge_prob: float, subgraph_size: int, 
                                    network_size: int, 
                                    cache: Dict = {}) -> float:
    """Calculate probability that a subgraph is connected (with memoization).
    
    Parameters
    ----------
    edge_prob : float
        Edge probability in parent graph
    subgraph_size : int
        Number of nodes in subgraph
    network_size : int
        Number of nodes in parent graph
    cache : Dict
        Dictionary for caching computed values
        
    Returns
    -------
    float
        Probability that the subgraph is connected
    """
    if edge_prob in cache:
        if network_size in cache[edge_prob]:
            if subgraph_size in cache[edge_prob][network_size]:
                return cache[edge_prob][network_size][subgraph_size]
    
    if subgraph_size == 1:
        return 1.0
    
    total = 0.0
    for k in range(1, subgraph_size):
        total += (connectedness_probability(edge_prob, k, network_size, cache) * 
                 comb(subgraph_size - 1, k - 1) * 
                 (1 - edge_prob) ** (k * (subgraph_size - k)))
    
    return float(1 - total)


def isolation_probability(edge_prob: float, subgraph_size: int, 
                                 network_size: int) -> float:
    """Calculate probability that selected nodes have no external neighbors.
    
    Parameters
    ----------
    edge_prob : float
        Edge probability in parent graph
    subgraph_size : int
        Number of selected nodes
    network_size : int
        Total number of nodes
        
    Returns
    -------
    float
        Probability of no external connections
    """
    return (1 - edge_prob) ** (subgraph_size * (network_size - subgraph_size))


def lcc_probability_raw(edge_prob: float, component_size: int, 
                                             network_size: int) -> float:
    """Calculate probability of largest component having specific size (no memoization).
    
    Parameters
    ----------
    edge_prob : float
        Edge probability
    component_size : int
        Size of largest component
    network_size : int
        Total number of nodes
        
    Returns
    -------
    float
        Probability of largest component having specified size
    """
    if component_size == 1 and network_size == 1:
        return 1.0
    elif component_size == 1 and network_size != 1:
        output = (1 - edge_prob) ** comb(network_size, 2) 
        return float(output)
    
    total = 0.0
    for j in range(0, component_size + 1):
        weight = 0.5 if j == component_size else 1.0
        total += weight * lcc_probability_raw(edge_prob, j, 
                                                                    network_size - component_size)
    
    output = (comb(network_size, component_size) * 
            connectedness_probability_raw(edge_prob, component_size, network_size) * 
            isolation_probability(edge_prob, component_size, network_size) * 
            total) 

    return float(output)


def lcc_probability(edge_prob: float, component_size: int, 
                                         network_size: int,
                                         connectivity_cache: Dict = {}, 
                                         probability_cache: Dict = {}) -> float:
    """Calculate probability of largest component having specific size (with memoization).
    
    Parameters
    ----------
    edge_prob : float
        Edge probability
    component_size : int
        Size of largest component  
    network_size : int
        Total number of nodes
    connectivity_cache : Dict
        Cache for connectivity probabilities
    probability_cache : Dict
        Cache for component probabilities
        
    Returns
    -------
    float
        Probability of largest component having specified size
    """
    if edge_prob in probability_cache:
        if network_size in probability_cache[edge_prob]:
            if component_size in probability_cache[edge_prob][network_size]:
                return probability_cache[edge_prob][network_size][component_size]
    
    if component_size == 1 and network_size == 1:
        return 1.0
    elif component_size == 1 and network_size != 1:
        output = (1 - edge_prob) ** comb(network_size, 2) 
        return float(output)
    
    total = 0.0
    for j in range(1, component_size + 1):
        weight = 0.5 if j == component_size else 1.0
        total += weight * lcc_probability(edge_prob, j, 
                                                               network_size - component_size,
                                                               connectivity_cache, 
                                                               probability_cache)
        
    output = (comb(network_size, component_size) * 
            connectedness_probability(edge_prob, component_size, network_size, 
                                           connectivity_cache) *
            isolation_probability(edge_prob, component_size, network_size) * 
            total)
    
    return float(output)


def lcc_probability_external(
        edge_prob: float, 
        component_size: int,
        network_size: int,
        executable_name: str = "p-recursion.exe") -> float:
    """Calculate probability using external executable.
    
    Parameters
    ----------
    edge_prob : float
        Edge probability
    component_size : int
        Component size
    network_size : int
        Network size
    executable_name : str
        Name of external executable program located in `repository_root/cpp`
        
    Returns
    -------
    float
        Probability from external calculation
    """
    executable_path = os.path.join(CPP_PATH, executable_name)
    output = execute_subprocess([executable_path, str(edge_prob), 
                                str(component_size), str(network_size)])
    return float(output) if output else 0.0


def expected_lcc_size_raw(edge_prob: float, 
                                               network_size: int) -> float:
    """Calculate expected largest component size (no memoization).
    
    Parameters
    ----------
    edge_prob : float
        Edge probability
    network_size : int
        Number of nodes
        
    Returns
    -------
    float
        Expected size of largest component
    """
    expected_size = 0.0
    for k in range(1, network_size + 1):
        expected_size += lcc_probability_raw(edge_prob, k, 
                                                                   network_size) * k
    return expected_size


def expected_lcc_size(edge_prob: float, network_size: int,
    connectivity_cache: Dict = {}, probability_cache: Dict = {},
    method: str = "internal",
    executable_name: str = "p-recursion.exe") -> float:
    """Calculate expected largest component size.
    
    Parameters
    ----------
    edge_prob : float
        Edge probability
    network_size : int
        Number of nodes
    connectivity_cache : Dict
        Cache for connectivity probabilities
    probability_cache : Dict
        Cache for component probabilities
    method : str
        Calculation method ('internal' or 'external')
    executable_name : str
        Name of external executable program located in `repository_root/cpp`
        to be used if method='external'
        
    Returns
    -------
    float
        Expected size of largest component
    """
    expected_size = 0.0
    
    if method == "external":
        for m in range(1, network_size + 1):
            expected_size += m * lcc_probability_external(
                edge_prob, m, network_size, executable_name)
    else:
        for m in range(1, network_size + 1):
            expected_size += m * lcc_probability(
                edge_prob, m, network_size, connectivity_cache, probability_cache)
    
    return expected_size


def lcc_sequence(edge_prob: float, network_size: int,
                             targeted_removal: bool = False,
                             reverse: bool = False,
                             connectivity_cache: Dict = {},
                             probability_cache: Dict = {},
                             method: str = "external",
                             executable_name: str = "p-recursion.exe") -> np.ndarray:
    """Calculate expected largest component sizes under sequential node removal.
    
    Parameters
    ----------
    edge_prob : float
        Initial edge probability
    network_size : int
        Initial network size
    targeted_removal : bool
        If True, remove nodes by degree; if False, remove randomly
    reverse : bool
        If True, return sizes in reverse order
    connectivity_cache : Dict
        Cache for connectivity probabilities
    probability_cache : Dict
        Cache for component probabilities
    method : str
        Calculation method
    executable_name : str
        Name of external executable program located in `repository_root/cpp`
        
    Returns
    -------
    np.ndarray
        Sequence of expected largest component sizes
    """
    sizes = np.zeros(network_size)
    current_prob = edge_prob
    
    for i in range(network_size - 1, -1, -1):
        sizes[i] = expected_lcc_size(
            current_prob, i + 1, connectivity_cache, probability_cache, 
            method, executable_name)
        
        if targeted_removal:
            current_prob = edge_probability_after_attack(i + 1, current_prob)
    
    if reverse:
        sizes = sizes[::-1]
    
    return sizes


def relative_lcc_sequence(edge_prob: float, network_size: int,
                                      targeted_removal: bool = False,
                                      reverse: bool = True,
                                      connectivity_cache: Dict = {},
                                      probability_cache: Dict = {},
                                      method: str = "internal",
                                      executable_name: str = "p-recursion.exe") -> np.ndarray:
    """Calculate relative largest component sizes under sequential node removal.
    
    Parameters
    ----------
    edge_prob : float
        Initial edge probability
    network_size : int
        Initial network size
    targeted_removal : bool
        If True, remove nodes by degree; if False, remove randomly
    reverse : bool
        If True, return sizes in reverse order
    connectivity_cache : Dict
        Cache for connectivity probabilities
    probability_cache : Dict
        Cache for component probabilities
    method : str
        Calculation method
    executable_name : str
        Name of external executable program located in `repository_root/cpp`
        
    Returns
    -------
    np.ndarray
        Sequence of relative largest component sizes
    """
    network_sizes = np.arange(1, network_size + 1)
    
    if reverse:
        network_sizes = network_sizes[::-1]
    
    absolute_sizes = lcc_sequence(
        edge_prob, network_size, targeted_removal, reverse,
        connectivity_cache, probability_cache, method, executable_name)
    
    return absolute_sizes / network_sizes


def relative_lcc_points(edge_prob: float = 0.1,
    network_sizes: List[int] = [20, 50, 100],
    connectivity_cache: Dict = {},
    probability_cache: Dict = {}) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate expected largest component sizes for specific network sizes.
    
    Parameters
    ----------
    edge_prob : float
        Edge probability
    network_sizes : List[int]
        List of network sizes to evaluate
    targeted_removal : bool
        If True, use targeted attack
    reverse : bool
        If True, reverse order
    connectivity_cache : Dict
        Cache for connectivity probabilities
    probability_cache : Dict
        Cache for component probabilities
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Network sizes and corresponding expected largest component sizes
    """
    sizes = np.array([
        expected_lcc_size(
            edge_prob, n, connectivity_cache, probability_cache)
        for n in network_sizes
    ])
    
    return np.array(network_sizes), sizes