"""
Network Robustness Visualization
Compares the effect of random vs targeted node removal on network connectivity.
Generates a two-panel figure showing simulation results alongside theoretical predictions.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add custom library path
sys.path.insert(0, "libs")

# Import custom modules for simulations and theory
from libs.robustnessSimulations import completeRCData
from libs.infiniteTheory import relSCurve

# ============= Configuration Parameters =============
NETWORK_SIZE = 20  # Number of nodes in the network
EDGE_PROBABILITY = 0.1  # Probability of edge existence in ER graphs
NUM_SIMULATIONS = 100  # Number of simulation trials for averaging
PROBABILITY_INDEX = int(EDGE_PROBABILITY / 0.01 - 1)  # Index for probability lookup

# ============= Main Figure Generation =============
def generate_robustness_comparison_figure():
    """
    Generate a two-panel figure comparing random vs targeted attack strategies
    on network robustness, showing simulations and theoretical predictions.
    """
    
    # Initialize figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3))
    
    # Define removal strategies to compare
    removal_strategies = [
        {'is_targeted': False, 'label': 'Random', 'strategy_name': 'random'},
        {'is_targeted': True, 'label': 'Target', 'strategy_name': 'attack'}
    ]
    
    # Generate x-axis values (fraction of nodes removed)
    node_fractions = np.arange(NETWORK_SIZE) / NETWORK_SIZE
    
    # Process each removal strategy
    for panel_idx, strategy in enumerate(removal_strategies):
        
        # ============= Run Simulations =============
        simulation_results = run_robustness_simulations(
            strategy['strategy_name'], 
            NETWORK_SIZE, 
            EDGE_PROBABILITY, 
            NUM_SIMULATIONS
        )
        
        # Calculate statistics for error bars
        mean_values, standard_errors = calculate_simulation_statistics(
            simulation_results, 
            NETWORK_SIZE, 
            NUM_SIMULATIONS
        )
        
        # ============= Load Theoretical Predictions =============
        # Infinite network theory
        infinite_theory = relSCurve(
            NETWORK_SIZE, 
            EDGE_PROBABILITY, 
            attack=strategy['is_targeted'], 
            smooth_end=False
        )
        
        # Finite network theory (pre-computed)
        finite_theory = load_finite_theory(
            strategy['is_targeted'], 
            NETWORK_SIZE, 
            PROBABILITY_INDEX
        )
        
        # ============= Plot Results =============
        plot_panel(
            axes[panel_idx], 
            node_fractions, 
            mean_values, 
            standard_errors, 
            infinite_theory, 
            finite_theory,
            show_ylabel=(panel_idx == 0),
            show_legend=(panel_idx == 1)
        )
        
        # Add panel labels
        axes[panel_idx].text(
            0.05, 0.9, f'({chr(97 + panel_idx)})', 
            transform=axes[panel_idx].transAxes, 
            fontsize=10, va='bottom', ha='left'
        )
    
    # Customize legend for the second panel
    customize_legend(axes[1])
    
    # Adjust layout and save
    plt.subplots_adjust(left=0.08, right=0.98, bottom=0.15, top=0.99, wspace=0.1)
    plt.savefig("fig_lcc_over_f_small.pdf")


def run_robustness_simulations(strategy, network_size, edge_prob, num_trials):
    """
    Run multiple simulation trials for a given removal strategy.
    
    Args:
        strategy: 'random' or 'attack' removal strategy
        network_size: Number of nodes in the network
        edge_prob: Edge probability for ER graphs
        num_trials: Number of simulation runs
    
    Returns:
        Array of simulation results for each trial
    """
    simulation_data = np.zeros(num_trials, dtype=object)
    
    for trial in range(num_trials):
        # Run single simulation
        sim_result = completeRCData(
            numbers_of_nodes=[network_size],
            edge_probabilities=[edge_prob], 
            num_trials=1,
            performance='relative LCC',  # Largest Connected Component
            graph_types=['ER'],  # Erdős-Rényi graphs
            remove_strategies=[strategy]
        )
        
        # Extract data array and remove header row
        data_array = np.array(sim_result[0][0][0][0])[1:]
        
        # Store mean across realizations
        simulation_data[trial] = np.nanmean(data_array, axis=0)
    
    return simulation_data


def calculate_simulation_statistics(sim_data, network_size, num_trials):
    """
    Calculate mean values and standard errors from simulation data.
    
    Args:
        sim_data: Array of simulation results
        network_size: Number of nodes in the network
        num_trials: Number of simulation runs
    
    Returns:
        Tuple of (mean_values, standard_errors)
    """
    mean_values = np.zeros(network_size)
    standard_errors = np.zeros(network_size)
    
    for node_idx in range(network_size):
        # Extract values at this removal fraction across all trials
        values_at_fraction = np.array([sim_data[trial][node_idx] for trial in range(num_trials)])
        
        # Calculate statistics
        mean_values[node_idx] = np.nanmean(values_at_fraction)
        standard_errors[node_idx] = np.std(values_at_fraction) / 10 * 3  # Standard error scaling
    
    return mean_values, standard_errors


def load_finite_theory(is_targeted, network_size, prob_index):
    """
    Load pre-computed finite network theory results.
    
    Args:
        is_targeted: Boolean indicating if targeted attack
        network_size: Number of nodes
        prob_index: Index for probability value
    
    Returns:
        Array of theoretical predictions
    """
    filename = f"RelSCurve_attack{is_targeted}_n{network_size}.npy"
    filepath = os.path.join("data", "synthetic_data", filename)
    all_theory = np.load(filepath)
    return all_theory[prob_index]


def plot_panel(ax, x_vals, sim_means, sim_errors, inf_theory, fin_theory, 
               show_ylabel=True, show_legend=False):
    """
    Plot simulation results and theoretical predictions on a single panel.
    
    Args:
        ax: Matplotlib axis object
        x_vals: Fraction of nodes removed
        sim_means: Mean simulation values
        sim_errors: Standard errors for error bars
        inf_theory: Infinite network theory predictions
        fin_theory: Finite network theory predictions
        show_ylabel: Whether to show y-axis label
        show_legend: Whether to show legend
    """
    # Plot simulation results with error bars
    ax.errorbar(
        x=x_vals, y=sim_means, yerr=sim_errors, 
        marker='o', markersize=2.5, label=r"$\widebar{S}$", 
        lw=1, color="red"
    )
    
    # Plot infinite network theory
    ax.plot(x_vals, inf_theory, label=r"${S}_{\infty}$", color="black")
    
    # Plot finite network theory
    ax.plot(x_vals, fin_theory, label=r"${S}_{rec}$", color="blue", linestyle='--')
    
    # Set axis labels
    ax.set(xlabel=r'fraction $f$')
    if show_ylabel:
        ax.set(ylabel=r'rel. LCC size')
    else:
        ax.set_yticklabels([])
    
    if show_legend:
        ax.legend()


def customize_legend(ax):
    """
    Customize the legend appearance and ordering.
    
    Args:
        ax: Matplotlib axis object with legend
    """
    # Reorder legend items
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [2, 0, 1]  # Reorder: Finite theory, Simulations, Infinite theory
    plt.legend(
        [handles[idx] for idx in order],
        [labels[idx] for idx in order],
        loc='upper left', 
        bbox_to_anchor=(0.6, 1)
    )


# ============= Execute Main Function =============
if __name__ == "__main__":
    generate_robustness_comparison_figure()