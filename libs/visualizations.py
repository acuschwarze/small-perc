###############################################################################
#
# Library of functions to plot network robustness curves.
#
# This library contains the following functions:
#     plot_graphs
###############################################################################

import numpy as np
import scipy.stats as sst
import networkx as nx
import matplotlib.pyplot as plt
from random import choice
from scipy.special import comb
from data import *
import infiniteTheory
import finiteTheory
from performanceMeasures import *
from robustnessSimulations import *
from scipy.signal import argrelextrema

# WORKS
def plot_graphs(numbers_of_nodes=[100], edge_probabilities=[0.1],
    graph_types=['ER', 'SF'], remove_strategies=['random', 'attack'],
    performance='largest_connected_component', num_trials=100,
    smooth_end=False, forbidden_values=[], fdict={}, pdict={}, lcc_method_main = "pmult", savefig='', simbool = True):
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

    num_trials : int (default=10)
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
       If savefig is a non-empty string, save of copy of the figure to that
       destination.
       
    Returns
    -------
    figure (a matplotlib figure)
       Figure with one or several subplots showing results.
    '''

    # get simulation data
    sim_data = completeRCData(numbers_of_nodes=numbers_of_nodes,
        edge_probabilities=edge_probabilities, num_trials=num_trials,
        performance=performance, graph_types=graph_types,
        remove_strategies=remove_strategies)

    # get number of graph types
    n_gt = len(graph_types)
    
    # get number of removal strategies
    n_rs = len(remove_strategies)

    # create a figure
    fig = plt.figure(figsize=(8,8))
    
    # select colors
    num_lines = len(numbers_of_nodes)*len(edge_probabilities)
    colors = plt.cm.jet(np.linspace(0,1,num_lines))

    # plot performance as function of the number of nodes removed
    ax_index = 0
    for i_gt, graph_type in enumerate(graph_types):
        for i_rs, remove_strategy in enumerate(remove_strategies):

            # create a subplot
            ax1 = plt.subplot(n_gt, n_rs, 1+ax_index)
            line_index = 0

            # create a new line for each combination of number of nodes and
            # number of edges
            for i_nn, n in enumerate(numbers_of_nodes):
                print(n)
                for i_ep, p in enumerate(edge_probabilities):
                    print(p)
                    # get relevant slice of simulated data
                    data_array = np.array(sim_data[i_gt][i_nn][i_ep][i_rs])
                    # exclude the first row, because it is the number of nodes
                    data_array = data_array[1:]

                    #this can prevent some sort of bug about invalid values
                    for val in forbidden_values:
                        data_array[data_array == val] = np.nan
                        
                    # plot simulated data
                    removed_fraction = np.arange(n)/n
                    line_data = np.nanmean(data_array,axis=0)

                    if simbool:
                        ax1.plot(removed_fraction, line_data,
                            'o', color=colors[line_index],
                            label="n={} , p={}".format(n , p))

                    if performance=='relative LCC':

                        attack = (remove_strategy=='attack')

                        # get data from finite theory
                        finiteRelS = finiteTheory.relSCurve(p,n,
                            attack=attack, fdict=fdict,pdict=pdict,lcc_method_relS=lcc_method_main)
                        print(finiteRelS)
                        # plot data from finite theory
                        ax1.plot(removed_fraction, finiteRelS,
                            color=colors[line_index],
                            label="finite th.")

                        # get data from infinite theory
                        infiniteRelS = infiniteTheory.relSCurve(n, p,
                            attack=attack, smooth_end=smooth_end)
  
                        # plot data from infinite theory
                        ax1.plot(removed_fraction, infiniteRelS,
                            ls='--', color=colors[line_index],
                            label="infinite th.")
                        print(removed_fraction,"sim x")
                    elif performance == "average small component size":

                        # get data from infinite theory
                        infiniteRelS = infiniteTheory.relSmallSCurve(p,n,
                            attack=attack, smooth_end=smooth_end)
                        
                        # plot data from infinite theory
                        ax1.plot(removed_fraction, infiniteRelS,
                            ls='--', color=colors[line_index],
                            label="infinite th.")
                        plt.ylim(0, 5)
                        
                    line_index += 1

            # label the plot
            ax1.set_title(str(performance) + " of " + str(graph_type) +
                          " graph, " + str(remove_strategy) + " removal")
            ax1.legend()
            ax1.set_xlabel('n (number nodes removed)')
            ax1.set_ylabel(performance)

    #plt.subplots_adjust(left=None, bottom=None, right=2, top=2, wspace=None, hspace=None)

    if len(savefig) > 0:
        plt.savefig(savefig)

    return fig