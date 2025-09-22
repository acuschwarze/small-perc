"""
Visual recap of percolation results and their applicability to finite networks
==============================================================================
Script to generate Intro_Figure.pdf comparing simulated and theoretical LCC sizes.
The output figure is a plot comparing relative LCC sizes for different edge probabilities.

The output figure is saved to 'repository root/figures/fig_intro.pdf'
"""

# Import libraries
import os, sys, pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.special import comb, lambertw
from pathlib import Path

# Add the parent directory to the path to import local libraries
REPO_ROOT = str(Path(__file__).parent.parent)
FIGURE_PATH = os.path.join(REPO_ROOT, 'figures')
CACHE_PATH = os.path.join(REPO_ROOT, 'cache-combinatorics')
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Import from local libraries
from libs.infiniteTheory import lambert_w_safe

# Load precomputed data
fvals = pickle.load(open(os.path.join(CACHE_PATH, 'fvalues.p'), 'rb'))
pvals = pickle.load(open(os.path.join(CACHE_PATH, 'Pvalues.p'), 'rb'))


def lambert_w_with_interpolation(x: float, k: int = 0, tol: float = 1E-20) -> complex:
    """
    Lambert-W function with interpolation close to the jump point of its zero-th branch.
    
    Parameters
    ----------
    x : float
        Argument of the Lambert-W function.
    k : int (default=0)
        Branch of the Lambert-W function.
    tol : float (default=1E-20)
        Tolerance for detecting proximity to jump point.
    
    Returns
    -------
    Union[float, Complex]
        Value of the Lambert-W function (with interpolation near jump point)
    """
    if np.abs(x + 1 / np.exp(1)) < tol:
        # If input is close to percolation threshold, set output to -1.0
        return -1.0
    else:
        return lambertw(x, k=k)


# Network parameters
network_size = 10
edge_probabilities = np.arange(51) / 100

# Initialize arrays for theoretical infinite network and simulation results
theoretical_infinite_lcc = np.zeros(len(edge_probabilities))
simulated_lcc = np.zeros(len(edge_probabilities))
error_bars = np.zeros(len(edge_probabilities))

# Calculate theoretical values for infinite networks
for i, prob in enumerate(edge_probabilities):
    mean_degree = 2 * prob * comb(network_size, 2) / network_size
    
    if mean_degree == 1 and network_size == 2:
        theoretical_infinite_lcc[i] = 2 / network_size
    elif mean_degree > 0:
        theoretical_infinite_lcc[i] = 1 + np.real(
            lambert_w_with_interpolation(-mean_degree * np.exp(-mean_degree), k=0, tol=1e-8) / mean_degree
        )
    else:
        theoretical_infinite_lcc[i] = 0

# Run simulations and calculate statistics
num_simulations = 100
for i, prob in enumerate(edge_probabilities):
    lcc_sizes = np.zeros(num_simulations)
    
    for j in range(num_simulations):
        graph = nx.erdos_renyi_graph(network_size, prob)
        largest_component_size = len(max(nx.connected_components(graph), key=len))
        lcc_sizes[j] = largest_component_size / network_size
    
    simulated_lcc[i] = np.mean(lcc_sizes)
    error_bars[i] = np.std(lcc_sizes) * 0.3  # Scaled standard deviation for error bars

# Create figure
fig, ax = plt.subplots(1, 1, figsize=[5, 3.5])

# Plot simulation results with error bars
plt.errorbar(x=edge_probabilities, y=simulated_lcc, yerr=error_bars, 
             marker='o', markersize=2.5, label=r"$\widebar{S}$", lw=1, color="red")

# Plot theoretical infinite network results
plt.plot(edge_probabilities, theoretical_infinite_lcc, 
         label=r"${S}_{\infty}$", color="black")

# Configure plot
plt.xlabel("edge probability " + r"$p$")
plt.ylabel("rel. LCC size")
plt.legend()

# Reorder legend entries
handles, labels = plt.gca().get_legend_handles_labels()
order = [1, 0]
plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

# Adjust layout and save
plt.subplots_adjust(left=0.12, right=.99, bottom=.15, top=0.99, wspace=0)
plt.savefig(os.path.join(FIGURE_PATH, "fig_intro.pdf"))