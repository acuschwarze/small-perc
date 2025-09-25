"""
Script to generate network percolation plots comparing simulations with theoretical predictions.

Output figures are saved to 'repository root/figures/fig_lcc_over_f_large-{REMOVAL_STRATEGY}.pdf'.

"""

# Import libraries
import os, sys
import pickle
import numpy as np
from typing import List
import matplotlib.pyplot as plt
from pathlib import Path

# Add the parent directory to the path to import local libraries
REPO_ROOT = str(Path(__file__).parent.parent)
FIGURE_PATH = os.path.join(REPO_ROOT, 'figures')
CACHE_PATH = os.path.join(REPO_ROOT, 'cache-combinatorics')
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Import from local libraries
import libs.finiteTheory as finiteTheory
import libs.infiniteTheory as infiniteTheory
from libs.utils import load_percolation_curve
from libs.robustnessSimulations import robustness_sweep


# Load precomputed data
fvals = pickle.load(open(os.path.join(CACHE_PATH, 'fvalues.p'), 'rb'))
pvals = pickle.load(open(os.path.join(CACHE_PATH, 'Pvalues.p'), 'rb'))

# Configuration parameters
NETWORK_THRESHOLD: float = 0.2
NETWORK_SIZES: List[int] = [10, 15, 25, 50]
EDGE_PROBABILITIES: List[float] = [1 / (NETWORK_THRESHOLD * (n - 1)) for n in NETWORK_SIZES]
NUM_SIMULATION_TRIALS: int = 100
USE_TARGETED_REMOVAL: bool = True  # True for attack, False for random
REMOVAL_STRATEGY: str = "attack" if USE_TARGETED_REMOVAL else 'random'

# Visualization settings
PLOT_COLORS: List[str] = ['red', 'blue', 'orange', 'green', 'purple', 'cyan', 'magenta']
PLOT_MARKERS: List[str] = ['o', 'x', 'v', 's', '+', 'd', '1']

for REMOVAL_STRATEGY in ['attack', 'random']:

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=[10, 3.5])

    # ============================================================================
    # SUBPLOT 1: Fixed network size with varying edge probabilities
    # ============================================================================
    fixed_network_size: int = 25
    probability_values: List[float] = [0.05, 0.1, 0.3, 1]
    node_fractions = np.arange(fixed_network_size) / fixed_network_size

    for prob_idx, edge_prob in enumerate(probability_values):
        # Calculate finite theory curve
        finite_theory_curve = load_percolation_curve(
            fixed_network_size, edge_prob, 
            targeted_removal=USE_TARGETED_REMOVAL,
            simulated=False, finite=True
        )[:fixed_network_size]
        
        # Generate simulation data with error bars
        simulation_data = np.zeros((NUM_SIMULATION_TRIALS, fixed_network_size))
        
        if edge_prob <= 0.1:
            # Run individual simulations for sparse networks
            for trial_idx in range(NUM_SIMULATION_TRIALS):
                sim_result = robustness_sweep(
                    numbers_of_nodes=[fixed_network_size],
                    edge_probabilities=[edge_prob], 
                    num_trials=1,
                    performance='relative LCC', 
                    graph_types=['ER'],
                    remove_strategies=[REMOVAL_STRATEGY]
                )
                trial_data = np.array(sim_result[0][0][0][0])[1:]  # Skip first row
                simulation_data[trial_idx] = np.nan_to_num(trial_data)
            
            mean_simulation = np.mean(simulation_data, axis=0)
            std_error = np.std(simulation_data, axis=0) / np.sqrt(NUM_SIMULATION_TRIALS) * 3
            
            # Plot with error bars
            axes[0].errorbar(
                node_fractions, mean_simulation, yerr=std_error,
                marker=PLOT_MARKERS[prob_idx], markersize=3, linestyle=' ',
                linewidth=1, color=PLOT_COLORS[prob_idx], label="_nolegend_"
            )
            axes[0].plot(
                node_fractions, mean_simulation,
                linestyle=' ', marker=PLOT_MARKERS[prob_idx], 
                color=PLOT_COLORS[prob_idx],
                label=fr"$p = {edge_prob}$", markersize=3
            )
        else:
            # Use precalculated simulations for dense networks
            precalc_sims = load_percolation_curve(
                fixed_network_size, edge_prob, 
                targeted_removal=USE_TARGETED_REMOVAL,
                simulated=True, finite=False
            )
            mean_simulation = np.mean(np.transpose(precalc_sims), axis=0)
            std_error = np.std(np.transpose(precalc_sims), axis=0) / np.sqrt(NUM_SIMULATION_TRIALS) * 3
            
            # Plot with error bars
            axes[0].errorbar(
                node_fractions, mean_simulation, yerr=std_error,
                marker=PLOT_MARKERS[prob_idx], markersize=3, linestyle=' ',
                linewidth=1, color=PLOT_COLORS[prob_idx], label="_nolegend_"
            )
            axes[0].plot(
                node_fractions, mean_simulation,
                linestyle=' ', marker=PLOT_MARKERS[prob_idx], 
                color=PLOT_COLORS[prob_idx],
                label=fr"$p = {edge_prob}$", markersize=3
            )
        
        # Plot infinite theory curve
        infinite_theory_curve = infiniteTheory.relative_lcc_sequence(
            fixed_network_size, edge_prob, 
            targeted_removal=USE_TARGETED_REMOVAL,
            smooth_end=False
        )
        axes[0].plot(node_fractions, infinite_theory_curve, color=PLOT_COLORS[prob_idx])
        
        # Plot finite theory curve (dashed)
        axes[0].plot(node_fractions, finite_theory_curve, linestyle='--', color=PLOT_COLORS[prob_idx])

    # Format subplot 1
    axes[0].set(xlabel="fraction " + r'$f$', ylabel='rel. LCC size')
    axes[0].set_ylim(-0.1, 1.05)
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4)

    # ============================================================================
    # SUBPLOT 2: Varying network sizes with fixed percolation threshold
    # ============================================================================
    for size_idx, network_size in enumerate(NETWORK_SIZES):
        node_fractions = np.arange(network_size) / network_size
        edge_prob = round(EDGE_PROBABILITIES[size_idx], 2)
        simulation_data = np.zeros((NUM_SIMULATION_TRIALS, network_size))
        
        # Run simulations for all trials
        for trial_idx in range(NUM_SIMULATION_TRIALS):
            sim_result = robustness_sweep(
                numbers_of_nodes=[network_size],
                edge_probabilities=[edge_prob], 
                num_trials=1,
                performance='relative LCC', 
                graph_types=['ER'],
                remove_strategies=[REMOVAL_STRATEGY]
            )
            trial_data = np.array(sim_result[0][0][0][0])[1:]  # Skip first row
            simulation_data[trial_idx] = np.nan_to_num(trial_data)
        
        # Calculate mean and standard error
        mean_simulation = np.nanmean(simulation_data, axis=0)
        std_error = np.nanstd(simulation_data, axis=0) / np.sqrt(NUM_SIMULATION_TRIALS) * 3
        
        # Plot simulation with error bars
        axes[1].errorbar(
            node_fractions, mean_simulation, yerr=std_error,
            marker=PLOT_MARKERS[size_idx], markersize=3, lw=1,
            color=PLOT_COLORS[size_idx], label="_nolegend_"
        )
        axes[1].plot(
            node_fractions, mean_simulation,
            linestyle=' ', marker=PLOT_MARKERS[size_idx], 
            color=PLOT_COLORS[size_idx],
            label=f"$N$={network_size}", markersize=3
        )
        
        # Plot finite theory curve
        if network_size > 100:
            finite_theory_curve = finiteTheory.relative_lcc_sequence(
                edge_prob, network_size, 
                targeted_removal=USE_TARGETED_REMOVAL, 
                connectivity_cache=fvals, probability_cache=pvals, 
                method="external",
                executable_name='p-recursion-float128.exe'
            )
        else:
            finite_theory_curve = load_percolation_curve(
                network_size, edge_prob, 
                targeted_removal=USE_TARGETED_REMOVAL,
                simulated=False, finite=True
            )[:network_size]
        
        axes[1].plot(node_fractions, finite_theory_curve, linestyle='--', color=PLOT_COLORS[size_idx])
        
        # Plot infinite theory curve (only for the last network size)
        if size_idx == len(NETWORK_SIZES) - 1:
            infinite_theory_curve = infiniteTheory.relative_lcc_sequence(
                network_size, edge_prob, 
                targeted_removal=USE_TARGETED_REMOVAL,
                smooth_end=False
            )
            axes[1].plot(node_fractions, infinite_theory_curve, color="black", label=r"$S_{\infty}$")

    # Format subplot 2
    axes[1].set_yticklabels([])
    axes[1].set_ylim(-0.1, 1.05)
    axes[1].set(xlabel=r'fraction $f$')
    axes[1].legend()

    # Add subplot labels
    axes[0].text(0.07, 0.1, '(a)', transform=axes[0].transAxes, 
                fontsize=10, fontweight='normal', va='top', ha='right')
    axes[1].text(0.07, 0.1, '(b)', transform=axes[1].transAxes, 
                fontsize=10, fontweight='normal', va='top', ha='right')

    # Adjust layout
    plt.subplots_adjust(left=0.06, right=0.99, bottom=0.15, top=0.90, wspace=0.04)

    # Save figures based on removal strategy
    plt.savefig(os.path.join(FIGURE_PATH, f"fig_lcc_over_f_large-{REMOVAL_STRATEGY}.pdf"))
