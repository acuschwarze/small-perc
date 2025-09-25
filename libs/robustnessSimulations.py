###############################################################################
#
# Library of functions to run computational node-removal experiments.
#
# Functions:
#     robustness_sequence - Run single node-removal experiment and track graph property changes
#     robustness_sequence_set - Run multiple experiments on sampled graphs from random-graph ensembles  
#     robustness_sweep - Run comprehensive experiments across multiple parameters and graph types
#     random_removal - Create LCC sequence under random node removal from a graph
#     targeted_removal - Create LCC sequence under degree-based targeted node removal from a graph
#     read_from_adj - Create graph from adjacency list file in Petter Holme's format
#     fullDataTable - Create table of percolation results for real networks
#     create_list_of_real_networks - Generate list of network files from specified directory
#
###############################################################################

import os, sys, time
from pathlib import Path
import scipy
import numpy as np
import pandas as pd
import networkx as nx
from typing import Tuple, List, Optional, Union
from random import choice
from fnmatch import fnmatch

# Add the parent directory to the path to import local libraries
REPO_ROOT = str(Path(__file__).parent.parent)
REAL_DATA_PATH = os.path.join(REPO_ROOT, 'data-real')
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Import from local libraries
from libs.performanceMeasures import *
from libs.finiteTheory import relative_lcc_sequence
from libs.utils import sample_network


def robustness_sequence(graph: nx.Graph, 
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
        'largest_connected_component': lambda g: size_of_lcc(g),
        'relative LCC': lambda g: relative_size_of_lcc(g),
        'average cluster size': lambda g: average_component_size(g),
        'average small component size': lambda g: average_small_component_size(g),
        'mean shortest path': lambda g: mean_shortest_path_length(g),
        'efficiency': lambda g: efficiency(g),
        'entropy': lambda g: entropy(g),
        'reachability': lambda g: reachability(g),
        'transitivity': lambda g: nx.transitivity(g),
        'resistance distance': lambda g: resistance_distance(g),
        'natural connectivity': lambda g: mean_communicability(g)
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
            node_to_remove = sorted(graph.degree, key=lambda x: x[1], reverse=True)[0][0] 
        else:
            raise ValueError(f'Unknown node removal strategy: {remove_nodes}')
            
        graph.remove_node(node_to_remove)

    return performance_data


def robustness_sequence_set(n: int = 100,
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
    percolation_threshold = 0

    for trial_idx in range(num_trials):
        sample_graph = sample_network(n, p, graph_type=graph_type)
        avg_degree = average_degree(sample_graph)
        
        percolation_threshold += 0 if avg_degree == 0 else 1 / avg_degree

        curve_data = robustness_sequence(sample_graph, 
                                     remove_nodes=remove_nodes,
                                     performance=performance)
        trial_data[trial_idx + 1] = curve_data[1]

    percolation_threshold /= num_trials

    return trial_data, percolation_threshold 


def robustness_sweep(numbers_of_nodes: List[int] = [100],
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
                    experiment_data = robustness_sequence_set(n=node_count,
                                             p=edge_prob,
                                             num_trials=num_trials,
                                             graph_type=graph_type,
                                             remove_nodes=strategy,
                                             performance=performance)[0]
                    results[graph_idx][node_idx][edge_idx][strategy_idx] = np.copy(experiment_data) 
                    
    return results 


def random_removal(G0: nx.Graph) -> np.ndarray:
    '''Create array of sequential LCC values when removing nodes uniformly at random from a graph.

    Parameters
    ----------
    G0 : nx.Graph
        Graph to remove nodes from.

    Returns
    -------
    np.ndarray
        Sequence of the LCC values of graph G0 under sequential random node removal.
    '''
    
    # make a copy of input graph
    G = G0.copy()
    n = G.number_of_nodes()
    
    data_array = np.zeros(n, dtype=float)
    
    for i in range(n):
        # get LCC size
        data_array[i] = len(max(nx.connected_components(G), key=len)) / (n - i)
        # find a random node to remove
        if G.number_of_nodes() != 0:
            v = choice(list(G.nodes()))
            G.remove_node(v)
            
    return data_array

            
def targeted_removal(G0: nx.Graph) -> np.ndarray:
    '''Create array of sequential LCC values when removing the highest degree nodes from a graph.

    Parameters
    ----------
    G0 : nx.Graph
        Graph to remove nodes from.

    Returns
    -------
    np.ndarray
        Sequence of the LCC values of graph G0 under sequential degree-targeted node removal.
    '''
        
    # make a copy of input graph
    G = G0.copy()
    n = G.number_of_nodes()
    
    data_array = np.zeros(n, dtype=float)
    for i in range(n):
        # get LCC size
        data_array[i] = len(max(nx.connected_components(G), key=len)) / (n - i)
        # find highest-degree node and remove it
        if G.number_of_nodes() != 0:
            v = sorted(G.degree, key=lambda x: x[1], reverse=True)[0][0]
            G.remove_node(v)
            
    return data_array


def read_from_adj(filename: str) -> nx.Graph:
    '''Create graph from the adjacency list within one of Petter Holme's network files.

    Parameters
    ----------
    filename : str
        Name of file with network adjacency list.

    Returns
    -------
    nx.Graph
        Graph representation of network.
    '''    
    file = open(filename, "r")
    content = file.readlines()

    # convert into networkx graph
    node_list = []
    edge_list = [] #np.empty(len(content), dtype=object)
    
    if len(content) == 0:
        G = nx.Graph()
        return G
    
    edge_count = 0
    for i in range(len(content)):
        
        edge = content[i].strip()
        edge = edge.split(" ")
        
        if len(edge)==2:
            
            edge_list.append([int(edge[0]), int(edge[1])])
            node_list.append(int(edge[0]))
            node_list.append(int(edge[1]))

    node_list = list(set(node_list))
    
    if 0 in node_list:
        n = max(node_list) + 1
        offset = 0
    else:
        n = max(node_list)
        offset = min(node_list)
    
    # create adjacency list
    adj = np.zeros((n, n))
        
    for k in range(len(edge_list)):
        adj[int(edge_list[k][0])-offset, int(edge_list[k][1])-offset] = 1
        adj[int(edge_list[k][1])-offset, int(edge_list[k][0])-offset] = 1

    G = nx.from_numpy_array(adj)
    file.close()
            
    return G


def fullDataTable(nwks_list: Optional[List[str]] = None, 
                  num_tries: int = 100, 
                  max_size: int = 100, 
                  min_counter: int = 0,
                  recompute=False) -> pd.DataFrame:
    '''Create table of percolation results values for real networks.

    Parameters
    ----------
    nwks_list : Optional[List[str]]
        Array of names of files with network data. If None, uses create_list_of_real_networks().
    
    num_tries : int, default=100
        Number of times to sample node removals from each network.

    max_size : int, default=100
        Maximum size of network to include in table creation.

    min_counter : int, default=0
        Minimum counter value to start processing networks (for resuming interrupted runs).
        
    Returns
    -------
    pd.DataFrame
        Dataframe of network properties and AUC values for targeted and random node removals,
        using sampled removals and the finite theory model.
    '''

    if nwks_list == None:
        nwks_list = create_list_of_real_networks()

    table = np.zeros((len(nwks_list),7), dtype=object)
    
    counter = 0
    nw_r = np.array([])
    nw_t = np.array([])
    theory_r = np.array([])
    theory_t = np.array([])
    for i, nwpath in enumerate(nwks_list):
        
        # extract file name from file path
        nwfile = os.path.basename(nwpath)
        nwname = nwfile.split('.')[0]
        print(f'{i}({counter}): {nwfile}', end='')
        # add name of network to table
        table[counter,0] =  str(nwfile)
        # search for existing data
        fpath = os.path.join(REAL_DATA_PATH, 'fulldata-{}.txt'.format(nwname))

        if os.path.exists(fpath) and not recompute:
            with open(fpath, 'r') as file:
                header = (file.readline()).split('\n')[0]
                _, n, m = header.split(" ")
                nw_r = (file.readline()).split('\n')[0]
                nw_t = (file.readline()).split('\n')[0]
                theory_r = (file.readline()).split('\n')[0]
                theory_t = (file.readline()).split('\n')[0]
                print (' --- load from file')

        else:
            # read graph from ".adj" file
            print('{} {}'.format(i, nwfile), end='')
            G = read_from_adj(nwpath)
            # set p for G(n,p) graph
            n = G.number_of_nodes()
            m = G.number_of_edges()
            print(' has (n,m) = ({}, {})'.format(n, m), end='')
            p = m / scipy.special.comb(n, 2)
        
            # check if network meets size limitation
            if n > max_size:
                print (' --- omit')
                continue
            elif n < 2:
                print(' --- omit')
                continue
            else:
                print(' --- compute', end='')

            if counter >= min_counter:
                t0 = time.time()
                # add number of nodes and edges to info table
                table[counter,1] = n
                table[counter,2] = m
        
                # get data for random and targeted node removal 
                nw_r = np.nanmean([random_removal(G) for i in range(num_tries)], axis=0)
                nw_t = targeted_removal(G)
                
                # finite-theory results for random and targeted node removal
                theory_r = relative_lcc_sequence(p, n, targeted_removal=False, 
                                                 reverse=True, method="external")
                theory_t = relative_lcc_sequence(p, n, targeted_removal=True, 
                                                 reverse=True, method="external")

                with open(fpath, 'w') as file:
                    # Write four lines to the file
                    file.write("{} {} {}\n".format(nwfile, n, m))
                    file.write(' '.join(map(str, nw_r))+"\n")
                    file.write(' '.join(map(str, nw_t))+"\n")
                    file.write(' '.join(map(str, theory_r))+"\n")
                    file.write(' '.join(map(str, theory_t))+"\n")

                print(' in {} s'.format(time.time()-t0))

        # rel LCC arrays
        results = [nw_r, nw_t, theory_r, theory_t]
        for j, array in enumerate(results): 
            # store in info table
            table[counter,3+j] = array
    
        counter+=1

    # remove empty rows from table
    table2 = table[:counter]

    # convert to data frame and name its columns
    df = pd.DataFrame(table2)
    df.columns = ["network", "nodes", "edges", "real rand rLCC", 
                  "real attack rLCC", "fin theory rand rLCC", 
                  "fin theory attack rLCC"]
    
    return df


def create_list_of_real_networks(folder_name: str = 'pholme_networks', 
                                  pattern: str = "*.adj") -> List[str]:
    '''Generate a list of network files from a specified directory.

    Parameters
    ----------
    folder_name : str, default='pholme_networks'
        Name of the folder containing network files, relative to REPO_ROOT.
    
    pattern : str, default="*.adj"
        File pattern to match when searching for network files.

    Returns
    -------
    List[str]
        List of full paths to network files matching the specified pattern.
    '''

    folder = os.path.join(REPO_ROOT, folder_name)
    nwks_list = []

    for path, subdirs, files in os.walk(folder):
        for name in files:
            if fnmatch(name, pattern):
                nwks_list.append(os.path.join(path, name))
            # elif fnmatch(name, pattern2):
            #     nwks_list2.append(os.path.join(path, name)

    return nwks_list