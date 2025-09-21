"""
Library for calculating theoretical percolation results for finite networks.

Functions:
    execute_subprocess: Execute an external program and return its output
    compute_connectivity_probability_raw: Calculate probability that a subgraph is connected (no memoization)
    compute_connectivity_probability: Calculate probability that a subgraph is connected (with memoization)
    compute_isolation_probability: Calculate probability that nodes have no external neighbors
    compute_largest_component_probability_raw: Calculate probability of largest component size (no memoization)
    compute_largest_component_probability: Calculate probability of largest component size (with memoization)
    compute_largest_component_probability_external: Calculate probability using external executable
    compute_expected_largest_component_size_raw: Calculate expected largest component size (no memoization)
    compute_expected_largest_component_size: Calculate expected largest component size (with memoization)
    compute_percolation_curve: Calculate expected largest component sizes under sequential node removal
    compute_relative_percolation_curve: Calculate relative largest component sizes under sequential node removal
    compute_percolation_points: Calculate expected largest component sizes for specific network sizes
    update_edge_probability_after_attack: Update edge probability after targeted node removal
"""

import numpy as np
import scipy.special
from scipy.special import comb
from typing import Dict, List, Tuple, Optional, Union
import subprocess
import math


def execute_subprocess(executable_path: List[str]) -> Optional[str]:
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
        return None


def compute_connectivity_probability_raw(edge_prob: float, subgraph_size: int, 
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
        total += (compute_connectivity_probability_raw(edge_prob, k, network_size) * 
                 comb(subgraph_size - 1, k - 1) * 
                 (1 - edge_prob) ** (k * (subgraph_size - k)))
    
    return 1 - total # type: ignore


def compute_connectivity_probability(edge_prob: float, subgraph_size: int, 
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
        total += (compute_connectivity_probability(edge_prob, k, network_size, cache) * 
                 comb(subgraph_size - 1, k - 1) * 
                 (1 - edge_prob) ** (k * (subgraph_size - k)))
    
    return 1 - total # type: ignore


def compute_isolation_probability(edge_prob: float, subgraph_size: int, 
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


def compute_largest_component_probability_raw(edge_prob: float, component_size: int, 
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
        return (1 - edge_prob) ** comb(network_size, 2) # type: ignore
    
    total = 0.0
    for j in range(0, component_size + 1):
        weight = 0.5 if j == component_size else 1.0
        total += weight * compute_largest_component_probability_raw(edge_prob, j, 
                                                                    network_size - component_size)
    
    return (comb(network_size, component_size) * 
            compute_connectivity_probability_raw(edge_prob, component_size, network_size) * 
            compute_isolation_probability(edge_prob, component_size, network_size) * 
            total) # type: ignore


def compute_largest_component_probability(edge_prob: float, component_size: int, 
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
        return (1 - edge_prob) ** comb(network_size, 2) # type: ignore
    
    total = 0.0
    for j in range(1, component_size + 1):
        weight = 0.5 if j == component_size else 1.0
        total += weight * compute_largest_component_probability(edge_prob, j, 
                                                               network_size - component_size,
                                                               connectivity_cache, 
                                                               probability_cache)
    
    return (comb(network_size, component_size) * 
            compute_connectivity_probability(edge_prob, component_size, network_size, 
                                           connectivity_cache) *
            compute_isolation_probability(edge_prob, component_size, network_size) * 
            total) # type: ignore


def compute_largest_component_probability_external(edge_prob: float, component_size: int,
                                                  network_size: int,
                                                  executable_path: str = "p-recursion.exe") -> float:
    """Calculate probability using external executable.
    
    Parameters
    ----------
    edge_prob : float
        Edge probability
    component_size : int
        Component size
    network_size : int
        Network size
    executable_path : str
        Path to external calculation program
        
    Returns
    -------
    float
        Probability from external calculation
    """
    output = execute_subprocess([executable_path, str(edge_prob), 
                                str(component_size), str(network_size)])
    return float(output) if output else 0.0


def compute_expected_largest_component_size_raw(edge_prob: float, 
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
        expected_size += compute_largest_component_probability_raw(edge_prob, k, 
                                                                   network_size) * k
    return expected_size


def compute_expected_largest_component_size(edge_prob: float, network_size: int,
                                           connectivity_cache: Dict = {},
                                           probability_cache: Dict = {},
                                           method: str = "internal",
                                           executable_path: str = "p-recursion.exe") -> float:
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
    executable_path : str
        Path to external program if method='external'
        
    Returns
    -------
    float
        Expected size of largest component
    """
    expected_size = 0.0
    
    if method == "external":
        for m in range(1, network_size + 1):
            expected_size += m * compute_largest_component_probability_external(
                edge_prob, m, network_size, executable_path)
    else:
        for m in range(1, network_size + 1):
            expected_size += m * compute_largest_component_probability(
                edge_prob, m, network_size, connectivity_cache, probability_cache)
    
    return expected_size


def update_edge_probability_after_attack(remaining_nodes: int, 
                                        current_edge_prob: float) -> float:
    """Update edge probability after targeted node removal.
    
    Parameters
    ----------
    remaining_nodes : int
        Number of remaining nodes
    current_edge_prob : float
        Current edge probability
        
    Returns
    -------
    float
        Updated edge probability
    """
    # Placeholder for actual implementation
    # This would contain the logic for updating probability after targeted attack
    return current_edge_prob


def compute_percolation_curve(edge_prob: float, network_size: int,
                             targeted_attack: bool = False,
                             reverse: bool = False,
                             connectivity_cache: Dict = {},
                             probability_cache: Dict = {},
                             method: str = "internal",
                             executable_path: str = "p-recursion.exe") -> np.ndarray:
    """Calculate expected largest component sizes under sequential node removal.
    
    Parameters
    ----------
    edge_prob : float
        Initial edge probability
    network_size : int
        Initial network size
    targeted_attack : bool
        If True, remove nodes by degree; if False, remove randomly
    reverse : bool
        If True, return sizes in reverse order
    connectivity_cache : Dict
        Cache for connectivity probabilities
    probability_cache : Dict
        Cache for component probabilities
    method : str
        Calculation method
    executable_path : str
        Path to external program
        
    Returns
    -------
    np.ndarray
        Sequence of expected largest component sizes
    """
    sizes = np.zeros(network_size)
    current_prob = edge_prob
    
    for i in range(network_size - 1, -1, -1):
        sizes[i] = compute_expected_largest_component_size(
            current_prob, i + 1, connectivity_cache, probability_cache, 
            method, executable_path)
        
        if targeted_attack:
            current_prob = update_edge_probability_after_attack(i + 1, current_prob)
    
    if reverse:
        sizes = sizes[::-1]
    
    return sizes


def compute_relative_percolation_curve(edge_prob: float, network_size: int,
                                      targeted_attack: bool = False,
                                      reverse: bool = True,
                                      connectivity_cache: Dict = {},
                                      probability_cache: Dict = {},
                                      method: str = "internal",
                                      executable_path: str = "p-recursion.exe") -> np.ndarray:
    """Calculate relative largest component sizes under sequential node removal.
    
    Parameters
    ----------
    edge_prob : float
        Initial edge probability
    network_size : int
        Initial network size
    targeted_attack : bool
        If True, remove nodes by degree; if False, remove randomly
    reverse : bool
        If True, return sizes in reverse order
    connectivity_cache : Dict
        Cache for connectivity probabilities
    probability_cache : Dict
        Cache for component probabilities
    method : str
        Calculation method
    executable_path : str
        Path to external program
        
    Returns
    -------
    np.ndarray
        Sequence of relative largest component sizes
    """
    network_sizes = np.arange(1, network_size + 1)
    
    if reverse:
        network_sizes = network_sizes[::-1]
    
    absolute_sizes = compute_percolation_curve(
        edge_prob, network_size, targeted_attack, reverse,
        connectivity_cache, probability_cache, method, executable_path)
    
    return absolute_sizes / network_sizes


def compute_percolation_points(edge_prob: float = 0.1,
                              network_sizes: List[int] = [20, 50, 100],
                              targeted_attack: bool = False,
                              reverse: bool = False,
                              connectivity_cache: Dict = {},
                              probability_cache: Dict = {}) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate expected largest component sizes for specific network sizes.
    
    Parameters
    ----------
    edge_prob : float
        Edge probability
    network_sizes : List[int]
        List of network sizes to evaluate
    targeted_attack : bool
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
        compute_expected_largest_component_size(
            edge_prob, n, connectivity_cache, probability_cache)
        for n in network_sizes
    ])
    
    return np.array(network_sizes), sizes