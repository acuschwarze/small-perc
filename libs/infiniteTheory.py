###############################################################################
#
# Library of functions to calculate theoretical percolation results for 
# infinite networks.
#
# Functions included:
#   - lambert_w_safe: Lambert-W function with interpolation near jump points
#   - relative_lcc_sequence: Expected relative sizes of largest connected
#                            component under sequential node removal
#   - small_components_sequence: Expected mean sizes of small connected
#                                components under sequential node removal
#
###############################################################################

# Import libraries
import sys
from pathlib import Path
import numpy as np
from typing import Tuple
from scipy.special import comb, lambertw

# Add the parent directory to the path to import local libraries
REPO_ROOT = str(Path(__file__).parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Import from local libraries
from libs.utils import edge_probability_after_attack
from libs.performanceMeasures import *


def lambert_w_safe(x: float, branch: int = 0, tolerance: float = 1E-20) -> float:
    """Lambert-W function with interpolation close to the jump point of its 
    zero-th branch.

    Parameters
    ----------
    x : float
        Argument of the Lambert-W function.
    branch : int (default=0)
        Branch of the Lambert-W function.
    tolerance : float (default=1E-20)
        Tolerance for detecting proximity to jump point.

    Returns
    -------
    float
        Value of the Lambert-W function (with interpolation near jump point).
    """
    if np.abs(x + 1 / np.exp(1)) < tolerance:
        return -1.0
    return lambertw(x, k=branch)


def relative_lcc_sequence(
    num_nodes: int, 
    edge_prob: float, 
    targeted_removal: bool = False, 
    smooth_end: bool = False
) -> np.ndarray:
    """Sequence of expected relative sizes of the largest connected component
    in an Erdos-Renyi random graph under sequential node removal.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the graph.
    edge_prob : float
        Edge probability in the graph.
    attack : bool (default=False)
        If True, target nodes by degree instead of uniformly at random.
    reverse : bool (default=False)
        If True, return expected sizes in reverse order.
    smooth_end : bool (default=False)
        If True, add inverse of current network size as lower bound.

    Returns
    -------
    np.ndarray
        Expected relative sizes of largest connected component.
    """
    relative_sizes = np.zeros(num_nodes)
    current_nodes = num_nodes
    current_edge_prob = edge_prob

    for i in range(num_nodes):
        mean_degree = 2 * current_edge_prob * comb(current_nodes, 2) / current_nodes

        if mean_degree == 1 and num_nodes == 2:
            relative_sizes[i] = 2 / current_nodes
        elif mean_degree > 0:
            relative_sizes[i] = 1 + np.real(
                lambert_w_safe(-mean_degree * np.exp(-mean_degree), branch=0, tolerance=1e-8) / mean_degree 
            )
        else:
            relative_sizes[i] = 0
            
        if smooth_end:
            relative_sizes[i] = max(relative_sizes[i], 1 / current_nodes)

        if targeted_removal:
            current_edge_prob = edge_probability_after_attack(current_nodes, current_edge_prob)

        if current_nodes > 1:
            current_nodes -= 1

    return relative_sizes


def small_components_sequence(
    num_nodes: int,
    edge_prob: float,
    targeted_removal: bool = False,
    smooth_end: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Sequence of expected mean sizes of small connected components
    in an Erdos-Renyi random graph under sequential node removal.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the graph.
    edge_prob : float
        Edge probability in the graph.
    attack : bool (default=False)
        If True, target nodes by degree instead of uniformly at random.
    smooth_end : bool (default=False)
        If True, apply smoothing at the end of the sequence.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Removed fraction and expected mean sizes of small components.
    """
    removed_fraction = np.arange(num_nodes) / num_nodes
    mean_sizes = np.zeros(num_nodes)
    current_nodes = num_nodes
    current_edge_prob = edge_prob

    for i in range(num_nodes):
        mean_degree = 2 * current_edge_prob * comb(current_nodes, 2) / current_nodes
        lambert_w_output = lambert_w_safe(-mean_degree * np.exp(-mean_degree),  
                                          branch=0, tolerance=1e-8)
        largest_component_size = 1 + lambert_w_output / mean_degree
        mean_sizes[i] = 1 / (1 - mean_degree + mean_degree * largest_component_size)

        if smooth_end:
            mean_sizes[i] = max(mean_sizes[i], 1 / current_nodes)

        if targeted_removal:
            current_edge_prob = edge_probability_after_attack(current_nodes, current_edge_prob)

        if current_nodes > 1:
            current_nodes -= 1

    return removed_fraction, mean_sizes