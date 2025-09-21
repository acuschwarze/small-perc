###############################################################################
#
# Library of functions to run computational node-removal experiments.
#
# Functions:
#     robustnessCurve - Run single node-removal experiment and track graph property changes
#     getRCSet - Run multiple experiments on sampled graphs from random-graph ensembles  
#     completeRCData - Run comprehensive experiments across multiple parameters and graph types
#
###############################################################################

import numpy as np
import networkx as nx
from typing import Tuple, List, Callable, Any
from random import choice
from data import *
from utils import *
from performanceMeasures import *


def robustnessCurve(graph: nx.Graph, 
                   remove_nodes: str = 'random',
                   performance: str = 'largest_connected_component') -> np.ndarray:
    '''Run a computational node-removal experiment on a graph and record 
    how a structural property changes as nodes are removed sequentially.

    Parameters
    ----------
    graph : networkx.Graph
        Input graph for the experiment.
    
    remove_nodes : str
        Node removal strategy: 'random' or 'attack' (by degree).
       
    performance : str
        Structural property to track during node removal.

    Returns
    -------
    np.ndarray
        2xN array: first row is nodes removed, second row is performance values.
    '''
    num_nodes = graph.number_of_nodes()
    performance_data = np.zeros((2, num_nodes), dtype=float)
    performance_data[0] = np.arange(num_nodes)

    # Map performance measure names to functions
    performance_functions = {
        'number_of_nodes': lambda g: g.number_of_nodes(),
        'largest_connected_component': lambda g: sizeOfLCC(g),
        'relative LCC': lambda g: relativeSizeOfLCC(g),
        'average cluster size': lambda g: averageComponentSize(g),
        'average small component size': lambda g: averageSmallComponentSize(g),
        'mean shortest path': lambda g: meanShortestPathLength(g),
        'efficiency': lambda g: getEfficiency(g),
        'entropy': lambda g: getEntropy(g),
        'reachability': lambda g: getReachability(g),
        'transitivity': lambda g: nx.transitivity(g),
        'resistance distance': lambda g: resistanceDistance(g),
        'natural connectivity': lambda g: meanCommunicability(g)
    }
    
    if performance not in performance_functions:
        raise ValueError(f"Invalid performance measure: {performance}")
    
    compute_performance = performance_functions[performance]

    for step in range(num_nodes):
        performance_data[1, step] = compute_performance(graph)
        
        if step == num_nodes - 1:
            break

        if remove_nodes == 'random':
            node_to_remove = choice(list(graph.nodes()))
        elif remove_nodes == 'attack':
            node_to_remove = sorted(graph.degree, key=lambda x: x[1], reverse=True)[0][0] # type: ignore
        else:
            raise ValueError(f'Unknown node removal strategy: {remove_nodes}')
            
        graph.remove_node(node_to_remove)

    return performance_data


def getRCSet(n: int = 100,
            p: float = 0.1,
            num_trials: int = 10,
            graph_type: str = 'ER',
            remove_nodes: str = 'random',
            performance: str = 'largest_connected_component') -> Tuple[np.ndarray, float]:
    '''Run multiple node-removal experiments on sampled graphs from a 
    random-graph ensemble and record performance measure changes.

    Parameters
    ----------
    n : int
        Number of nodes in sampled networks.
    p : float
        Edge probability in sampled networks.
    num_trials : int
        Number of sample networks to generate.
    graph_type : str
        Graph model type: 'ER' or 'BA'.
    remove_nodes : str
        Node removal strategy: 'random' or 'attack'.
    performance : str
        Structural property to track.

    Returns
    -------
    Tuple[np.ndarray, float]
        Performance data array and percolation threshold.
    '''
    trial_data = np.zeros((num_trials + 1, n), dtype=float)
    trial_data[0] = np.arange(n)

    for trial_idx in range(num_trials):
        sample_graph = sampleNetwork(n, p, graph_type=graph_type)
        avg_degree = averageDegree(sample_graph)
        
        percolation_threshold = 0 if avg_degree == 0 else 1 / avg_degree

        curve_data = robustnessCurve(sample_graph, 
                                     remove_nodes=remove_nodes,
                                     performance=performance)
        trial_data[trial_idx + 1] = curve_data[1]

    return trial_data, percolation_threshold # type: ignore


def completeRCData(numbers_of_nodes: List[int] = [100],
                  edge_probabilities: List[float] = [0.1],
                  num_trials: int = 10,
                  performance: str = 'largest_connected_component',
                  graph_types: List[str] = ['ER', 'SF'],
                  remove_strategies: List[str] = ['random', 'attack']) -> List[List[List[List[np.ndarray]]]]:
    '''Run comprehensive node-removal experiments across multiple parameters,
    graph types, and removal strategies.

    Parameters
    ----------
    numbers_of_nodes : List[int]
        List of node counts for experiments.
    edge_probabilities : List[float]
        List of edge probabilities for experiments.
    num_trials : int
        Number of trials per parameter combination.
    performance : str
        Structural property to track.
    graph_types : List[str]
        Graph model types to test.
    remove_strategies : List[str]
        Node removal strategies to test.

    Returns
    -------
    List[List[List[List[np.ndarray]]]]
        Nested results indexed by: [graph_type][node_count][edge_prob][strategy].
    '''
    results = [[[[None for _ in range(len(remove_strategies))]
                for _ in range(len(edge_probabilities))]
               for _ in range(len(numbers_of_nodes))]
              for _ in range(len(graph_types))]

    for graph_idx, graph_type in enumerate(graph_types):
        for node_idx, node_count in enumerate(numbers_of_nodes):
            for edge_idx, edge_prob in enumerate(edge_probabilities):
                for strategy_idx, strategy in enumerate(remove_strategies):
                    experiment_data = getRCSet(n=node_count,
                                             p=edge_prob,
                                             num_trials=num_trials,
                                             graph_type=graph_type,
                                             remove_nodes=strategy,
                                             performance=performance)[0]
                    results[graph_idx][node_idx][edge_idx][strategy_idx] = np.copy(experiment_data) # type: ignore
                    
    return results # type: ignore