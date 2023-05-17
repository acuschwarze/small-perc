###############################################################################
#
# Library of functions to plot network robustness curves.
#
# This library contains the following functions:
#     plot_graphs
#
###############################################################################

import numpy as np
import scipy
import scipy.stats as sst
import networkx as nx
import matplotlib.pyplot as plt
from random import choice
from scipy.special import comb
from data import *


# WORKS
def plot_graphs(graph_types=['ER', 'SF'],
                numbers_of_nodes=[100],
                numbers_of_edges=[1, 2, 3],
                remove_strategies=['random', 'attack'],
                performance='efficiency', to_vary='nodes', vary_index=1, smoothing=False, both=False,
                forbidden_values=[]):  # figure out what vary_index did again
                
    LIST = completeRobustnessData(graph_types, numbers_of_nodes, numbers_of_edges, remove_strategies, performance)

    if to_vary == 'nodes': # I use this one mostly - it means that we're comparing by graph size
        # start plotting
        fig = plt.figure()

        # 1. plot performance as function of n
        x = 1
        while x < 2: # since only 1 subplot right now, set while x<2
            for i_gt, graph_type in enumerate(graph_types):
                for i_rs, remove_strategy in enumerate(remove_strategies):

                    ax1 = plt.subplot(2, 2, x)  # 1 subplot for every combination of graph type/rs
                                                # only 1 subplot right now
                    x += 1
                    print("x")
                    for i_nn, number_of_node in enumerate(numbers_of_nodes):
                        # percolation threshold stuff from the robustness graphs - vertical line
                        # if computeRobustnessCurves(graph_type=graph_type,
                        #                          number_of_nodes=number_of_node, remove_nodes = remove_strategy,
                        #                          number_of_edges = numbers_of_edges[0],
                        #                          performance=performance)[1] > 0:
                        #   percolation_threshold_graph = 1 - computeRobustnessCurves(graph_type=graph_type, remove_nodes = remove_strategy,
                        #                          number_of_nodes=number_of_node,
                        #                          number_of_edges = numbers_of_edges[0],
                        #                          performance=performance)[1]
                        #   print("pcg", 1-percolation_threshold_graph)
                        # generating plot
                        # x_points = computeRobustnessCurves(graph_type=graph_type,
                        #                          number_of_nodes=number_of_node,
                        #                          number_of_edges = numbers_of_edges[0],
                        #                          performance=performance)[2]

                        # calculating p from the m and n values in the parameters - when I put
                        # p=.1 in the construct_a_network function and below, this line gets overwritten
                        p = 2 * numbers_of_edges[i_nn] * (number_of_node - numbers_of_edges[i_nn]) / (
                                    number_of_node * (number_of_node - 1))

                        # don't know why we have this
                        calc_array = np.flip(np.arange(number_of_node) + 1)
                        #print("calc_array")
                        #calculated = big_S(p, np.flip(np.arange(number_of_node)),fvalues,pvalues)

                        # calculate the rel LCC from recursion for p=.1, and the graph size in question
                        calculated = big_relS(.1,[number_of_node],fvalues,pvalues)
                        print(calculated)
                        #print(calculated) # checkpoints
                        #print("calculated")
                        x_points = np.zeros(number_of_node)
                        for u in range(number_of_node):
                            x_points[u] = u / number_of_node
                        #print("xpoints")

                        #get simulated data from the big list
                        data_array = np.array(LIST[i_gt][i_nn][vary_index][i_rs][1:])
                        # print("data array shape", data_array.shape)

                        #this prevented some sort of bug about invalid values
                        for val in forbidden_values:
                            data_array[data_array == val] = np.nan
                        # print("data array shape", data_array.shape)

                        # print(np.nanmean(data_array,
                        # axis = 0))
                        #plot the simulated data
                        ax1.plot(x_points,  # range(numbers_of_nodes[i_nn]),
                                 np.nanmean(data_array,
                                            axis=0), 'o-',
                                 label="number of nodes=" + str(numbers_of_nodes[i_nn]))
                        print(np.nanmean(data_array,
                                            axis=0))
                        #print("nanmean")
                        # plt.show()

                        # simulation
                        #p = 2 * numbers_of_edges[0] * (numbers_of_nodes[0] - numbers_of_edges[0]) / (
                                    #numbers_of_nodes[0] * (numbers_of_nodes[0] - 1))

                        p=.1 # I guess I also defined p here too for the lambert function
                        if performance == "relative LCC" and remove_strategies == ["random"]:
                            # the commented lines are from other notebooks where I was plotting different things, like
                            # lambert functions while keeping the ratio of p to graph size constant, etc.

                            # ax1.plot(rich(p, fixed=.8)[0],rich(p, fixed=.8)[1])
                            # ax1.plot(perf_sim2copy(numbers_of_nodes[0], p, smoothing = smoothing)[0], perf_sim2copy(numbers_of_nodes[0], p, smoothing = smoothing)[1], label = 'Lambert Input')

                            #lambert plot
                            ax1.plot(perf_sim2copy(numbers_of_nodes[0], p, smoothing=smoothing)[0],
                                     perf_sim2copy(numbers_of_nodes[0], p, smoothing=smoothing)[2], label='Theoretical')
                            #recursive plot
                            ax1.plot(np.flip(np.arange(0,number_of_node))/number_of_node, calculated,
                                     label="calculated")
                            print(calculated) #checkpoint
                            plt.show()
                            ax1.legend()

                            # ax1.plot(perf_sim2copy(numbers_of_nodes[0], p, smoothing = smoothing)[0], perf_sim2copy(numbers_of_nodes[0], p, smoothing = smoothing)[3], label = 'mean degree')
                            # ax1.plot(perf_sim2copy(numbers_of_nodes[0], p, smoothing = smoothing)[0], perf_sim2copy(numbers_of_nodes[0], p, smoothing = smoothing)[4], label = 'difference')
                            # ax1.plot(np.linspace(0,1,100), -1/np.exp(1)*np.ones(shape=(100)))
                            # ax1.plot(np.linspace(0,1,100), np.ones(shape=(100)))
                            # plt.xlim(.4, .6)
                            # plt.ylim(-.5,1.2)

                            # ax1.plot(perf_sim_revision1(numbers_of_nodes[0], p)[0], perf_sim_revision1(numbers_of_nodes[0], p)[1])
                            print("work")
                        if performance == "relative LCC" and remove_strategies == ["attack"]:
                            ax1.plot(perf_sim2_attack(numbers_of_nodes[0], p, smoothing=smoothing)[0],
                                     perf_sim2_attack(numbers_of_nodes[0], p, smoothing=smoothing)[1])

                        if performance == "average small component size" and remove_strategies == ["random"]:
                            if both == True:
                                ax1.plot(perf_sim2(numbers_of_nodes[0], p, smoothing=smoothing)[0],
                                         perf_sim2(numbers_of_nodes[0], p, smoothing=smoothing)[1])
                            ax1.plot(s_sim(numbers_of_nodes[0], p)[0], s_sim(numbers_of_nodes[0], p)[1])
                            plt.ylim(0, 5)

                        if performance == "average small component size" and remove_strategies == ["attack"]:
                            ax1.plot(s_sim_attack(numbers_of_nodes[0], p)[0], s_sim_attack(numbers_of_nodes[0], p)[1])
                            plt.ylim(0, 5)

                        if performance == "both" and remove_strategies == ["random"]:
                            ax1.plot(perf_sim2(numbers_of_nodes[0], p, smoothing=smoothing)[0],
                                     perf_sim2(numbers_of_nodes[0], p, smoothing=smoothing)[1])
                            ax1.plot(s_sim(numbers_of_nodes[0], p)[0], s_sim(numbers_of_nodes[0], p)[1])
                            plt.ylim(0, 5)

                        # print(range(numbers_of_nodes[i_nn]))
                        # print(np.nanmean(LIST[i_gt][i_nn][vary_index][i_rs][1:], axis = 0))
                        # labeling plot
                        ax1.set_title(str(performance) + " of " + str(graph_type) +
                                      " graph, " + str(remove_strategy) + " removal: over "
                                      + to_vary)
                        ax1.legend()
                        ax1.set_xlabel('n (number nodes removed)')
                        ax1.set_ylabel(performance)

                        # this was from robustness library stuff/plotting different percolation thresholds
                        #for different sized graphs.
                        if number_of_node == numbers_of_nodes[0]:
                            set_color = "tab:blue"
                        # if number_of_node == numbers_of_nodes[1]:
                        #   set_color = "tab:orange"
                        # if number_of_node == numbers_of_nodes[2]:
                        #   set_color = "tab:green"
                        # if number_of_node == numbers_of_nodes[3]:
                        #   set_color = "tab:red"
                        # if number_of_node == numbers_of_nodes[4]:
                        #   set_color = "tab:purple"
                        #laterpercolation_threshold_graph = 1 - perf_sim2copy(numbers_of_nodes[0], p, smoothing=smoothing)[5]
                        # print(percolation_threshold_graph)
                        #laterax1.axvline(percolation_threshold_graph, color=set_color)

                    # fig.tight_layout()
            plt.subplots_adjust(left=None, bottom=None, right=2, top=2, wspace=None, hspace=None)
            plt.show()

    # this isn't used so much now - it was mostly from robustness stuff when I would compare
    # performance values when there were different densities
    elif to_vary == 'edges':
        # start plotting
        fig = plt.figure()

        # 1. plot performance as function of n, 2 * 2 * 5 figures
        x = 1
        while x < 5:
            for i_gt, graph_type in enumerate(graph_types):
                for i_rs, remove_strategy in enumerate(remove_strategies):
                    ax1 = plt.subplot(2, 2, x)  # 1 subplot for every combination of graph type/rs
                    x += 1

                    for i_ne, number_of_edge in enumerate(numbers_of_edges):
                        # generating plot
                        ax1.plot(range(numbers_of_nodes[1]),
                                 np.mean(LIST[i_gt][vary_index][i_ne][i_rs][1:],
                                         axis=0), 'o-',
                                 label="number of edges=" + str(numbers_of_edges[i_ne]))

                        # labeling plot
                        ax1.set_title(str(performance) + " of " + str(graph_type) +
                                      " graph, " + str(remove_strategy) + " removal: over "
                                      + to_vary)
                        ax1.legend()
                        ax1.set_xlabel('n (number nodes removed)')
                        ax1.set_ylabel(performance)
                        # plt.show
                    # fig.tight_layout()
            plt.subplots_adjust(left=None, bottom=None, right=2, top=2, wspace=None, hspace=None)
            plt.show()

        else:
            print("Please vary either nodes or edges.")
            # safety check if user made a mistake with to_vary

if __name__ == "__main__":
    #plot_graphs(['ER'], [2], [1], ["random"], performance = 'relative LCC', vary_index = 0, smoothing = True, forbidden_values = [0])
    #plt.show()
    plot_graphs(['ER'], [20], [1], ["random"], performance = 'relative LCC', vary_index = 0, smoothing = True, forbidden_values = [0])
    plt.show()
