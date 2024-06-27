###############################################################################
#
# Library of functions to run computational node-removal experiments.
#
# This library contains the following functions:
#     robustnessCurve (previously "computeRobustnessCurve")
#     getRCSet (previously "computeRobustnessCurves")
#     completeRCData(previously "completeRobustnessData")
#
###############################################################################

import numpy as np
import scipy
import scipy.stats as sst
import networkx as nx
from random import choice
from scipy.special import comb
from data import *
from utils import *
from performanceMeasures import *
#from Dictionaries import *

def robustnessCurve(g, remove_nodes='random', 
    performance='largest_connected_component'):
    '''Run a computational node-removal experiment on a graph `g` and record 
    how a structural property of that graph changes as one removes nodes 
    sequentially.
    
    ISSUE #1: What is going on with the smoothing code?

    Parameters
    ----------
    g : a networkX graph
       A graph.
    
    remove_nodes : str (default='random')
       If remove_nodes is 'random', select nodes to be removed uniformly at 
       random. If remove_nodes is 'attack', select nodse to be removed by
       largest degree. (AS: Adaptively?)
       
    performance : str (default='largest_connected_component')
       Structural property that is tracked over the node-removal experiment.
       Default is the number of nodes in the largest connected component. There
       are several options available.

    Returns
    -------
    data_array : 2D numpy array
       A 2xN array (where N is the number of nodes in g). The first row 
       reports the number of nodes removed. The second row reports the 
       corresponding value of the structural property (i.e., performance 
       measure).
    '''
    # get number of nodes
    n = g.number_of_nodes()

    # initialize data array: 2 rows of n columns each
    # row 1 is number of nodes removed, row 2 is performance measurement
    data_array = np.zeros((2, n), dtype=float)

    # set values for row 1
    data_array[0] = np.arange(n)

    # select performance measure
    if performance == 'number_of_nodes':
        computePerformance = lambda g: g.number_of_nodes()
    elif performance == 'largest_connected_component':
        computePerformance =  lambda g: sizeOfLCC(g)
    elif performance == "relative LCC":
        computePerformance =  lambda g: relativeSizeOfLCC(g)
    elif performance == "average cluster size":
        computePerformance = lambda g: averageComponentSize(g)
    elif performance == "average small component size":
        computePerformance = lambda g: averageSmallComponentSize(g)
    elif performance == "mean shortest path":
        computePerformance = lambda g: meanShortestPathLength(g)
    elif performance == 'efficiency':
        computePerformance = lambda g: getEfficiency(g)
    elif performance == "entropy":
        computePerformance = lambda g: getEntropy(g)
    elif performance == "reachability":
        computePerformance = lambda g: getReachability(g)
    elif performance == "transitivity":
        computePerformance = lambda g: nx.transitivity(g)
    elif performance == "resistance distance":
        computePerformance = lambda g: resistanceDistance(g)
    elif performance == "natural connectivity":
        computePerformance = lambda g: meanCommunicability(g)
    else:
        raise ValueError("Invalid performance value")

    # compute values of performance measure throughout node removal
    for i in range(n):
        # calculate performance value
        data_array[1, i] = computePerformance(g)
        
        if i == n:
            break

        # find a node to remove
        if remove_nodes == 'random':
            # select a random node
            v = choice(list(g.nodes()))
        elif remove_nodes == 'attack':
            # select the most connected node
            v = sorted(g.degree, key=lambda x: x[1], reverse=True)[0][0]
        else:
            raise ValueError('I dont know that mode of removing nodes')
            v = None
        # remove node
        g.remove_node(v)



        # if performance == "average small component size smooth": # AS: What is happening here?
        #   if i in np.arange(0, (1 / k) * g.number_of_nodes(), 1):
        #       data_array[1, i] = n / (nx.number_connected_components(g) - 1)

    return data_array


def getRCSet(n=100, p=0.1, num_trials=10, graph_type='ER',
    remove_nodes='random', performance='largest_connected_component'):
    '''Run several computational node-removal experiments on graphs sampled
    from a random-graph ensemble and record how a structural property of those
    graphs change as one removes nodes sequentially.

    ISSUE #1: Returns the same percolation threshold for all realizations? Does 
    that make sense?
    
    ISSUE #2: Computes two percolation thresholds but only returns one.
    
    ISSUE #3: Not clear to me that returning 0 for networks under percolation
    threhold makes sense.

    Parameters
    ----------
    n : int (default=100)
       Number of nodes in sampled networks.

    p : float (default=0.1)
       Edge probability in sampled networks.

    num_trials : int (default=10)
       Number of sample networks drawn from the random-graph model.
    
    graph_type : str (default='ER')
       If graph_type=='ER', use samples of the Erdos--Renyi random-graph model;
       if graph_type=='BA', use samples of the Barabasi--Albert model.

    remove_nodes : str (default='random')
       If remove_nodes is 'random', select nodes to be removed uniformly at 
       random. If remove_nodes is 'attack', select nodse to be removed by
       largest degree. (AS: Adaptively?)
       
    performance : str (default='largest_connected_component')
       Structural property that is tracked over the node-removal experiment.
       Default is the number of nodes in the largest connected component. There
       are several options available.

    Returns
    -------
    data_array : 2D numpy array
       A (num_trials+1)xN array (where N is the number of nodes in g). The 
       first row reports the number of nodes removed. The subsequent rows report
       the corresponding value of the structural property (i.e., performance
       measure) in each trial.
    
    percolation_threshold : float
       The percolation threshold for an Erdos--Renyi network with the 
       prescribed number of nodes and number of edges.
       (AS: Why do we return this here?)
    '''
    # array to store data from multiple trials
    data_array = np.zeros((num_trials + 1, n), dtype=float)
    data_array[0] = np.arange(n)

    for i in range(num_trials):
        g = sampleNetwork(n, p, graph_type=graph_type)
        c = averageDegree(g)
        if c == 0:
            percolation_threshold = 0 # AS: Is this the right choice? #JJ: I think it may be 1. above 1 has giant component, below has none
        else:
            #not sure why both are here? Maybe to calculate percolation with two different methods?
            # AS: One of these lines is redundant?
            # AS: It doesn't seem like this calculation has to happen in the loop is networks are generated correctly?
            percolation_threshold = 1 / c
            # i think this is incorrect percolation_threshold = 1 / n + (n - 1) / (c * n)

        # get robustness curve for graph g
        data = robustnessCurve(g, remove_nodes=remove_nodes, 
            performance=performance)

        # add performance data to data array
        data_array[i + 1] = data[1]

    return data_array, percolation_threshold


def completeRCData(numbers_of_nodes=[100], edge_probabilities=[0.1],
    num_trials=10, performance='largest_connected_component',
    graph_types=['ER', 'SF'], remove_strategies=['random', 'attack']):
    '''Run several computational node-removal experiments on graphs sampled
    from an Erdos--Renyi random-graph model and a Barabasi--Albert random-graph
    model and record how a structural property of those graphs change as one 
    removes nodes sequentially either uniformly at random or targeted by
    degree.

    Parameters
    ----------
    numbers_of_nodes : list (default=[100])
       Numbers of nodes in sampled networks.

    edge_probabilities : list (default=[0.1])
       Edge probabilities in sampled networks.

    num_trials : int (default=10)
       Number of sample networks drawn from each random-graph model for each
       combination of numbers of nodes and numbers of edges.

    performance : str (default='largest_connected_component')
       Structural property that is tracked over the node-removal experiment.
       Default is the number of nodes in the largest connected component. There
       are several options available.

    graph_types : list (default=['ER', 'SF'])
       When graph_type=='ER', use samples of the Erdos--Renyi random-graph 
       model; when graph_type=='BA', use samples of the Barabasi--Albert model.

    remove_strategies : list (default=['random', 'attack'])
       When remove_nodes is 'random', select nodes to be removed uniformly at
       random. When remove_nodes is 'attack', select nodse to be removed by
       largest degree. (AS: Adaptively?)

    Returns
    -------
    res : 2D numpy array
       Nested list of results. First index determines graph type, second index
       determines number of nodes, third index determines number of edges, 
       fourth index determines removal strategy.

    '''
    # initialize some big list
    res = [[[[0 for i in range(len(remove_strategies))]
              for j in range(len(edge_probabilities))]
             for k in range(len(numbers_of_nodes))]
            for l in range(len(graph_types))]

    for i_gt, graph_type in enumerate(graph_types):
        for i_nn, n in enumerate(numbers_of_nodes):
            for i_ep, p in enumerate(edge_probabilities):
                for i_rs, remove_strategy in enumerate(remove_strategies):
                    # for every graph size, average degree, and removal 
                    # strategy, get the data of the performance values
                    # over all the trials (10 is the default)
                    data = getRCSet(n=n, p=p, num_trials=num_trials,
                        graph_type=graph_type, remove_nodes=remove_strategy,
                        performance=performance)[0]

                    # collect data in list
                    res[i_gt][i_nn][i_ep][i_rs] = np.copy(data)
    return res
