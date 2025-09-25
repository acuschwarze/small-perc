###############################################################################
#
# Library of functions to plot network robustness curves.
#
# Functions included:
#     plot_graphs - Plots network robustness curves for various graph types,
#                   removal strategies, and performance measures. Compares
#                   simulation data with finite and infinite theory predictions.
#
###############################################################################

# Import libraries
import os, sys
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Add the parent directory to the path to import local libraries
REPO_ROOT = str(Path(__file__).parent.parent)
FIGURE_PATH = os.path.join(REPO_ROOT, 'figures')
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Import from local libraries
import libs.infiniteTheory as infiniteTheory
import libs.finiteTheory as finiteTheory
from libs.robustnessSimulations import robustness_sweep


def plot_graphs(
    numbers_of_nodes: List[int] = [100],
    edge_probabilities: List[float] = [0.1],
    graph_types: List[str] = ['ER', 'SF'],
    remove_strategies: List[str] = ['random', 'attack'],
    performance: str = 'relative LCC',
    num_trials: int = 100,
    smooth_end: bool = False,
    forbidden_values: List[float] = [],
    fdict: Dict[Any, Any] = {},
    pdict: Dict[Any, Any] = {},
    lcc_method_main: str = "external",
    savefig: str = '',
    simbool: bool = True,
    executable_recursion: str = "p-recursion.exe",
    executable_max_degree: str = "max-degree.exe"
) -> Figure:
    '''Calculate edge probability in an Erdos--Renyi network with original size
    `n` and original edge probability `p` after removing the node with the
    highest degree.

    Parameters
    ----------
    graph_types : list (default=['ER', 'SF'])
       List of random-graph models from which networks should be sampled.

    numbers_of_nodes : list (default=[100])
       List of initial network sizes.
       
    edge_probabilities : list (default=[0.1])
       List of initial edge probabilities.
       
    remove_strategies : list (default = ['random', 'attack'])
       List of removal strategies (either uniformly at random or by node degree
       for nodes and by sum of incident node degrees for edges).
       
    performance : str (default='largest_connected_component')
       Performance measure to be used.

    num_trials : int (default=100)
       Number of sample networks drawn from each random-graph model for each
       combination of numbers of nodes and numbers of edges.

    smooth_end : bool (default=False)
       If smooth_end is True, apply end smoothing for infinite-theory results.

    forbidden_values : list (default=[])
       List of values to exclude from the plot.
       
    fdict (default={})
       Dictionary of precomputed values of f.
       
    pdict (default={})
       Dictionary of precomputed values of P.

    lcc_method_main (default='abc')
       # TODO: Add description.
       
    savefig : str (default='')
       If savefig is a non-empty string, save of copy of the figure to 
       `repository_root/figures/savefig'. Remember to include a file ending 
       (e.g., '.png' or '.pdf').
       
    Returns
    -------
    figure (a matplotlib figure)
       Figure with one or several subplots showing results.
    '''

    # Get simulation data
    sim_data = robustness_sweep(
        numbers_of_nodes=numbers_of_nodes,
        edge_probabilities=edge_probabilities,
        num_trials=num_trials,
        performance=performance,
        graph_types=graph_types,
        remove_strategies=remove_strategies
    )

    num_graph_types = len(graph_types)
    num_removal_strategies = len(remove_strategies)

    # Create figure
    fig = plt.figure(figsize=(8, 8))
    
    # Select colors for plotting
    num_lines = len(numbers_of_nodes) * len(edge_probabilities)
    colors = plt.cm.jet(np.linspace(0, 1, num_lines)) 

    # Plot performance as function of the number of nodes removed
    subplot_index = 0
    for graph_type_idx, graph_type in enumerate(graph_types):
        for removal_idx, remove_strategy in enumerate(remove_strategies):

            # Create subplot
            subplot = plt.subplot(num_graph_types, num_removal_strategies, 1 + subplot_index)
            line_index = 0

            # Create line for each combination of nodes and edges
            for node_count_idx, node_count in enumerate(numbers_of_nodes):
                print(node_count)
                for edge_prob_idx, edge_prob in enumerate(edge_probabilities):
                    # Get relevant slice of simulated data (exclude first row - node count)
                    data_array = np.array(sim_data[graph_type_idx][node_count_idx][edge_prob_idx][removal_idx])
                    data_array = data_array[1:]

                    # Filter forbidden values
                    for val in forbidden_values:
                        data_array[data_array == val] = np.nan
                        
                    # Plot simulated data
                    removed_fraction = np.arange(node_count) / node_count
                    line_data = np.nanmean(data_array, axis=0)

                    if simbool:
                        subplot.plot(
                            removed_fraction, line_data,
                            'o', color=colors[line_index],
                            label=f"n={node_count}, p={edge_prob}"
                        )

                    is_attack = bool(remove_strategy == 'attack')
                    if performance == 'relative LCC':

                        # Get and plot finite theory data
                        finite_rel_s = finiteTheory.relative_lcc_sequence( 
                            edge_prob, node_count,
                            targeted_removal=is_attack,
                            connectivity_cache=fdict,
                            probability_cache=pdict,
                            method=lcc_method_main,
                            #executable_max_degree=executable_max_degree,
                            executable_name=executable_recursion
                        )
                        print(finite_rel_s)
                        subplot.plot(
                            removed_fraction, finite_rel_s,
                            color=colors[line_index],
                            label="finite th."
                        )

                        # Get and plot infinite theory data
                        infinite_rel_s = infiniteTheory.relative_lcc_sequence( 
                            node_count, edge_prob,
                            targeted_removal=is_attack,
                            smooth_end=smooth_end
                        )
                        subplot.plot(
                            removed_fraction, infinite_rel_s,
                            ls='--', color=colors[line_index],
                            label="infinite th."
                        )
                        print(removed_fraction, "sim x")

                    elif performance == "average small component size":
                        # Get and plot infinite theory data for small components
                        infinite_rel_s = infiniteTheory.small_components_sequence(
                            node_count, edge_prob, 
                            targeted_removal=is_attack, 
                            smooth_end=smooth_end 
                        )
                        subplot.plot(
                            removed_fraction, infinite_rel_s,
                            ls='--', color=colors[line_index],
                            label="infinite th."
                        )
                        plt.ylim(0, 5)
                        
                    line_index += 1

            # Label subplot
            subplot.set_title(
                f"{performance} of {graph_type} graph, {remove_strategy} removal"
            )
            subplot.legend()
            subplot.set_xlabel('n (number nodes removed)')
            subplot.set_ylabel(performance)

    if len(savefig) > 0:
        plt.savefig(os.path.join(FIGURE_PATH,savefig))

    return fig