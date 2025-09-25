"""
Visualization of N-p Grid of Network Robustness Results
=======================================================
This module visualizes network robustness results under random and targeted 
attacks, comparing simulated results with theoretical predictions for a grid of
various network sizes and edge probabilities.

Output figure is saved to 'repository root/figures/fig_heatmaps.pdf'.
"""

# Import libraries
import os
import sys
import pickle
import numpy as np
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
from matplotlib.colorbar import Colorbar
from matplotlib.collections import QuadMesh
from pathlib import Path
from typing import Dict, Optional, Any, List, Union, Tuple

# Add the parent directory to the path to import local libraries
REPO_ROOT = str(Path(__file__).parent.parent)
FIGURE_PATH = os.path.join(REPO_ROOT, 'figures')
FCACHE_PATH = os.path.join(REPO_ROOT, 'cache-figures')
CCACHE_PATH = os.path.join(REPO_ROOT, 'cache-combinatorics')
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Import from local libraries
from libs.utils import load_percolation_curve
import libs.infiniteTheory as infiniteTheory
#import libs.finiteTheory as finiteTheory

# Load precomputed values
fvals = pickle.load(open(os.path.join(CCACHE_PATH, 'fvalues.p'), 'rb'))
pvals = pickle.load(open(os.path.join(CCACHE_PATH, 'Pvalues.p'), 'rb'))

# Cache configuration
CACHE_FILE = Path(os.path.join(FCACHE_PATH, 'heatmap_data.pkl'))


def ensure_cache_dir() -> None:
    """Create cache directory if it doesn't exist."""
    Path(FCACHE_PATH).mkdir(exist_ok=True)


def load_cached_heatmaps(
    cache_file: Union[Path, str], 
    max_nodes: int, 
    total_probs: int
) -> Optional[Dict[str, Any]]:
    """
    Load cached heatmap data if available and parameters match.
    
    Parameters:
    -----------
    cache_file : Union[Path, str]
        Path to the cache file
    max_nodes : int
        Maximum number of nodes in the network
    total_probs : int
        Number of probability values to test
        
    Returns:
    --------
    Optional[Dict[str, Any]]
        Dictionary containing cached heatmap data, or None if cache invalid
    """
    cache_path = Path(cache_file) if isinstance(cache_file, str) else cache_file
    
    if not cache_path.exists():
        print("Cache file not found. Computing heatmaps from scratch...")
        return None
    
    try:
        with open(cache_path, 'rb') as f:
            cached_data = pickle.load(f)
            
        # Verify cache parameters match
        if (cached_data['max_nodes'] == max_nodes and 
            cached_data['total_probs'] == total_probs):
            print(f"Loading cached heatmap data from {cache_path}")
            return cached_data
        else:
            print("Cache parameters don't match. Recomputing heatmaps...")
            return None
    except Exception as e:
        print(f"Error loading cache: {e}. Recomputing heatmaps...")
        return None


def save_heatmaps(
    cache_file: Union[Path, str], 
    heatmap_data: Dict[str, Any]
) -> None:
    """
    Save computed heatmap data to cache file.
    
    Parameters:
    -----------
    cache_file : Union[Path, str]
        Path to save the cache file
    heatmap_data : Dict[str, Any]
        Dictionary containing heatmap data and parameters
    """
    ensure_cache_dir()
    cache_path = Path(cache_file) if isinstance(cache_file, str) else cache_file
    
    with open(cache_path, 'wb') as f:
        pickle.dump(heatmap_data, f)
    print(f"Heatmap data cached to {cache_path}")


def compute_robustness_heatmaps(
    max_nodes: int, 
    total_probs: int
) -> Dict[str, Any]:
    """
    Compute robustness heatmaps for random and targeted attacks.
    
    This function compares simulated network robustness curves with:
    - Finite network theoretical predictions
    - Infinite network theoretical predictions
    
    Parameters:
    -----------
    max_nodes : int
        Maximum number of nodes in the network
    total_probs : int
        Number of probability values to test
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing all computed heatmap arrays
    """
    nodes_array = np.arange(2, max_nodes + 1)
    probs_array = np.linspace(0.01, 1, total_probs)
    
    # Initialize heatmaps for random attacks (r) and targeted attacks (t)
    # auc: Area Under Curve, fin: finite theory, inf: infinite theory
    heatmap_random_auc = np.zeros((total_probs, len(nodes_array)))
    heatmap_random_mse_finite = np.zeros((total_probs, len(nodes_array)))
    heatmap_random_mse_infinite = np.zeros((total_probs, len(nodes_array)))
    
    heatmap_targeted_auc = np.zeros((total_probs, len(nodes_array)))
    heatmap_targeted_mse_finite = np.zeros((total_probs, len(nodes_array)))
    heatmap_targeted_mse_infinite = np.zeros((total_probs, len(nodes_array)))
    
    error_count = 0
    
    print("Computing robustness heatmaps...")
    for node_idx, num_nodes in enumerate(nodes_array):
        if num_nodes % 5 == 0:
            print(f"Processing node size {num_nodes}/{max_nodes}")
            
        for prob_idx, edge_prob in enumerate(probs_array):
            # Random attack simulations
            random_sim_all = load_percolation_curve(
                num_nodes, edge_prob, targeted_removal=False, 
                simulated=True, finite=False
            )
            random_sim = np.zeros(num_nodes)
            for k in range(num_nodes):
                random_sim += np.transpose(random_sim_all[:, k][:num_nodes])
            random_sim /= num_nodes
            
            # Random attack theoretical curves
            random_finite = load_percolation_curve(
                num_nodes, edge_prob, targeted_removal=False,
                simulated=False, finite=True
            )[:num_nodes]
            
            random_infinite = infiniteTheory.relative_lcc_sequence(
                num_nodes, edge_prob, targeted_removal=False, smooth_end=False
            )
            
            # Targeted attack simulations
            targeted_sim_all = load_percolation_curve(
                num_nodes, edge_prob, targeted_removal=True,
                simulated=True, finite=False
            )
            targeted_sim = np.zeros(num_nodes)
            for k in range(num_nodes):
                targeted_sim += np.transpose(targeted_sim_all[:, k][:num_nodes])
            targeted_sim /= num_nodes
            
            # Targeted attack theoretical curves
            targeted_finite = load_percolation_curve(
                num_nodes, edge_prob, targeted_removal=True,
                simulated=False, finite=True
            )[:num_nodes]
            
            targeted_infinite = infiniteTheory.relative_lcc_sequence(
                num_nodes, edge_prob, targeted_removal=True, smooth_end=False
            )
            
            # Calculate AUC (Area Under Curve)
            if num_nodes == 2:  # Special case for 2 nodes
                heatmap_random_auc[prob_idx][node_idx] = 1
                heatmap_targeted_auc[prob_idx][node_idx] = 1
            else:
                heatmap_random_auc[prob_idx][node_idx] = scipy.integrate.simpson(
                    random_sim, dx=1 / (num_nodes - 1)
                )
                heatmap_targeted_auc[prob_idx][node_idx] = scipy.integrate.simpson(
                    targeted_sim, dx=1 / (num_nodes - 1)
                )
            
            # Calculate MSE for random attacks
            mse_random_finite = ((random_finite - random_sim) ** 2).mean()
            if mse_random_finite == 0:
                heatmap_random_mse_finite[prob_idx][node_idx] = -7
            else:
                heatmap_random_mse_finite[prob_idx][node_idx] = np.log10(mse_random_finite)
            
            mse_random_infinite = ((random_infinite - random_sim) ** 2).mean()
            if mse_random_infinite == 0:
                heatmap_random_mse_infinite[prob_idx][node_idx] = -2
            else:
                heatmap_random_mse_infinite[prob_idx][node_idx] = np.log10(mse_random_infinite)
            
            # Calculate MSE for targeted attacks
            mse_targeted_finite = ((targeted_finite - targeted_sim) ** 2).mean()
            if mse_targeted_finite == 0:
                heatmap_targeted_mse_finite[prob_idx][node_idx] = -7
            elif mse_targeted_finite <= 10**(-7):
                heatmap_targeted_mse_finite[prob_idx][node_idx] = -7
                error_count += 1
            else:
                heatmap_targeted_mse_finite[prob_idx][node_idx] = np.log10(mse_targeted_finite)
            
            mse_targeted_infinite = ((targeted_infinite - targeted_sim) ** 2).mean()
            if mse_targeted_infinite == 0:
                heatmap_targeted_mse_infinite[prob_idx][node_idx] = -2
            else:
                heatmap_targeted_mse_infinite[prob_idx][node_idx] = np.log10(mse_targeted_infinite)
    
    print(f"Computation complete. Numerical errors encountered: {error_count}")
    
    return {
        'max_nodes': max_nodes,
        'total_probs': total_probs,
        'nodes_array': nodes_array,
        'probs_array': probs_array,
        'random_auc': heatmap_random_auc.tolist(),
        'random_mse_finite': heatmap_random_mse_finite.tolist(),
        'random_mse_infinite': heatmap_random_mse_infinite.tolist(),
        'targeted_auc': heatmap_targeted_auc.tolist(),
        'targeted_mse_finite': heatmap_targeted_mse_finite.tolist(),
        'targeted_mse_infinite': heatmap_targeted_mse_infinite.tolist()
    }


def add_colorbar_to_plot(
    mappable: QuadMesh, 
    use_log_scale: bool = False
) -> Colorbar:
    """
    Add a colorbar to the current plot.
    
    Parameters:
    -----------
    mappable : QuadMesh
        The plot element to add colorbar for
    use_log_scale : bool
        Whether to format colorbar labels as powers of 10
        
    Returns:
    --------
    Colorbar
        The created colorbar object
    """
    current_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure # type: ignore
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    colorbar = fig.colorbar(mappable, cax=cax) # type: ignore
    
    if use_log_scale:
        tick_labels = colorbar.ax.get_yticklabels()
        new_labels = [label.get_text().replace('−', '-') for label in tick_labels]
        formatted_labels = [r'$10^{{{}}}$'.format(x) for x in new_labels]
        colorbar.ax.set_yticklabels(formatted_labels)
    
    plt.sca(current_axes)
    return colorbar


def create_robustness_figure(
    heatmap_data: Dict[str, Any], 
    output_file: str = "fig_heatmaps.pdf"
) -> None:
    """
    Create the main robustness analysis figure with heatmaps and histograms.
    
    Parameters:
    -----------
    heatmap_data : Dict[str, Any]
        Dictionary containing all heatmap arrays and parameters
    output_file : str
        Path to save the output figure
    """
    # Extract data from dictionary
    nodes_array = heatmap_data['nodes_array']
    probs_array = heatmap_data['probs_array']
    
    # Create mesh grid for plotting
    x_nodes, y_probs = np.meshgrid(nodes_array, probs_array)
    
    # Flatten arrays for histograms
    hist_random_finite = np.ravel(heatmap_data['random_mse_finite'])
    hist_random_infinite = np.ravel(heatmap_data['random_mse_infinite'])
    hist_targeted_finite = np.ravel(heatmap_data['targeted_mse_finite'])
    hist_targeted_infinite = np.ravel(heatmap_data['targeted_mse_infinite'])
    
    # Create figure with subplots
    fig, axes = plt.subplots(
        nrows=2, ncols=5, figsize=[13, 6],
        gridspec_kw={"width_ratios": [1, 1, 1, 0.01, 1], "wspace": 0.3}
    )
    ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = axes
    
    plt.subplots_adjust(left=0.05, right=0.99, bottom=0.1, top=0.9)
    
    # Create custom colormap (using last 75% of gnuplot2)
    base_cmap = cm.get_cmap('gnuplot2')
    custom_cmap = cm.colors.ListedColormap(base_cmap(np.linspace(0.3, 1.0, 256))) # type: ignore
    reversed_cmap = custom_cmap.reversed()
    
    # Plot random attack results (top row)
    plot1 = ax1.pcolormesh(x_nodes, y_probs, heatmap_data['random_auc'])
    plot2 = ax2.pcolormesh(x_nodes, y_probs, heatmap_data['random_mse_finite'], 
                           cmap=reversed_cmap, vmin=-4, vmax=-0.3)
    plot3 = ax3.pcolormesh(x_nodes, y_probs, heatmap_data['random_mse_infinite'],
                           cmap=reversed_cmap, vmin=-4, vmax=-0.3)
    
    # Random attack histogram
    ax5.hist(hist_random_finite, density=True, label=r"${S}_{rec}$", alpha=0.65)
    ax5.hist(hist_random_infinite, density=True, label=r"${S}_{\infty}$", alpha=0.65)
    ax5.legend(prop={'size': 10})
    ax5.set_xlim([-7, 0.3])
    ax5.set_ylim([0, 1.3])
    
    # Plot targeted attack results (bottom row)
    plot6 = ax6.pcolormesh(x_nodes, y_probs, heatmap_data['targeted_auc'])
    plot7 = ax7.pcolormesh(x_nodes, y_probs, heatmap_data['targeted_mse_finite'],
                           cmap=reversed_cmap, vmin=-4, vmax=-0.3)
    plot8 = ax8.pcolormesh(x_nodes, y_probs, heatmap_data['targeted_mse_infinite'],
                           cmap=reversed_cmap, vmin=-4, vmax=-0.3)
    
    # Targeted attack histogram
    ax10.hist(hist_targeted_finite, density=True, label=r"${S}_{rec}$", alpha=0.65)
    ax10.hist(hist_targeted_infinite, density=True, label=r"${S}_{\infty}$", alpha=0.65)
    ax10.legend(prop={'size': 10})
    ax10.set_xlim([-7, 0.3])
    ax10.set_ylim([0, 1.3])
    
    # Configure axes visibility
    ax2.set_yticklabels([])
    ax3.set_yticklabels([])
    ax4.axis("off")
    ax7.set_yticklabels([])
    ax8.set_yticklabels([])
    ax9.axis("off")
    
    ax5.set_xticklabels([])
    ax1.set_xticklabels([])
    ax2.set_xticklabels([])
    ax3.set_xticklabels([])
    
    # Format x-axis labels for bottom histogram
    x_labels = [label.get_text().replace('−', '-') for label in ax10.get_xticklabels()]
    x_label_values = [int(a) for a in x_labels]
    formatted_labels = [r'$10^{{{}}}$'.format(x) for x in x_label_values]
    ax10.set_xticklabels(formatted_labels)
    
    # Add colorbars
    add_colorbar_to_plot(plot1)
    add_colorbar_to_plot(plot2, use_log_scale=True)
    add_colorbar_to_plot(plot3, use_log_scale=True)
    add_colorbar_to_plot(plot6)
    add_colorbar_to_plot(plot7, use_log_scale=True)
    add_colorbar_to_plot(plot8, use_log_scale=True)
    
    # Set labels
    ax1.set(ylabel=r'edge probability $p$')
    ax5.set(ylabel=r'$frequency$')
    ax6.set(xlabel=r'network size $N$', ylabel=r'edge probability $p$')
    ax7.set(xlabel=r'network size $N$')
    ax8.set(xlabel=r'network size $N$')
    ax10.set(xlabel=r'MSE', ylabel=r'$frequency$')
    
    # Set titles
    ax1.set_title(r"AUC of $\widebar{S}$")
    ax2.set_title(r"MSE of ${S}_{rec}$")
    ax3.set_title(r'MSE of ${S}_{\infty}$')
    ax5.set_title("Histogram of MSE")
    
    # Add subplot labels
    subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']
    for ax, label in zip([ax1, ax2, ax3, ax5, ax6, ax7, ax8, ax10], subplot_labels):
        ax.text(0.965, 0.965, label, transform=ax.transAxes, 
               fontsize=10, fontweight='normal', va='top', ha='right')
    
    fname = os.path.join(FIGURE_PATH, output_file)
    plt.savefig(fname)
    print(f"Figure saved to {fname}")


def generate_figure(
    max_nodes: int = 30, 
    total_probs: int = 100, 
    force_recompute: bool = False
) -> Dict[str, Any]:
    """
    Main function to generate Figure 3 with caching support.
    
    Parameters:
    -----------
    max_nodes : int
        Maximum number of nodes in the network
    total_probs : int
        Number of probability values to test
    force_recompute : bool
        If True, ignore cache and recompute all data
        
    Returns:
    --------
    Dict[str, Any]
        The heatmap data used for plotting
    """
    # Try to load cached data
    if not force_recompute:
        cached_data = load_cached_heatmaps(CACHE_FILE, max_nodes, total_probs)
        if cached_data is not None:
            create_robustness_figure(cached_data)
            return cached_data
    
    # Compute heatmaps from scratch
    heatmap_data = compute_robustness_heatmaps(max_nodes, total_probs)
    
    # Save to cache
    save_heatmaps(CACHE_FILE, heatmap_data)
    
    # Create figure
    create_robustness_figure(heatmap_data)
    
    return heatmap_data


if __name__ == "__main__":
    # Generate figure with default parameters
    # Set force_recompute=True to ignore cache and recalculate
    data = generate_figure(max_nodes=50, total_probs=100, force_recompute=True)