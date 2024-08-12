import sys, pickle
sys.path.insert(0, "libs")

import os, pickle, csv # import packages for file I/O
import time # package to help keep track of calculation time

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

import scipy
import scipy.stats as sst
from scipy.special import comb
from scipy.integrate import simpson
from scipy.signal import argrelextrema
from random import choice
import csv

from libs.utils import *
from libs.finiteTheory import *
from visualizations import *
from libs.utils import *
from robustnessSimulations import *
from performanceMeasures import *
from infiniteTheory import *
from finiteTheory import *

fvals = pickle.load(open('data/fvalues.p', 'rb'))
pvals = pickle.load(open('data/Pvalues.p', 'rb'))

nwks_list2 = pd.read_csv("nwks_list2")
print(len(nwks_list2))

print(nwks_list2.iloc[0])
cwd = os.getcwd() 
print("Current working directory:", cwd) 


def nodecount_edge(file_name = ""):
    file = open(file_name, "r")
    #content = file.readlines()
    content = (line.rstrip() for line in file)  # All lines including the blank ones
    content = list(line for line in content if line)
    print(content)
    node_list = []
    edge_list = np.empty(len(content), dtype=object)
    for i in range(len(content)):
        edge = content[i].strip()
        edge = edge.split(" ")
        edge_list[i] = np.zeros(2)
        print("i", i)
        print("edge[0]",edge[0])
        edge_list[i][0] = int(edge[0])
        edge_list[i][1] = int(edge[1])
        for j in range(2):
            node_list.append(int(edge[j]))
    if 0 in node_list:
        n = max(node_list) + 1
    else:
        n = max(node_list)
    edge = len(edge_list)
    return n,edge


def read_from_adj(filename):
    
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
        
    adj = np.zeros((n, n))
        
    for k in range(len(edge_list)):
        adj[int(edge_list[k][0])-offset, int(edge_list[k][1])-offset] = 1
        adj[int(edge_list[k][1])-offset, int(edge_list[k][0])-offset] = 1

    G = nx.from_numpy_array(adj)
    file.close()
            
    return G

    
def random_removal(G0):
    
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

            
def targeted_removal(G0):
    
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




def fullDataTable(nwks_list, num_tries=100, max_size=100, min_counter=0):

    table = np.zeros((len(nwks_list),7), dtype=object)
    
    counter = 0
    for i, nwpath in enumerate(nwks_list):
        
        # extract file name from file path
        nwname = os.path.basename(nwpath)
        # add name of network to table
        table[counter,0] =  str(nwname)
        # read graph from ".adj" file
        print('{} {}'.format(i, nwname), end='')
        G = read_from_adj(nwpath)
        # set p for G(n,p) graph
        n = G.number_of_nodes()
        m = G.number_of_edges()
        p = m / scipy.special.comb(n, 2)
        print(' has (n,m) = ({}, {})'.format(n, m), end='')
        
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
            theory_r = relSCurve(p, n, attack=False, reverse=True, lcc_method_relS="pmult")
            theory_t = relSCurve(p, n, attack=True, reverse=True, lcc_method_relS="pmult")
        
            # rel LCC arrays
            results = [nw_r, nw_t, theory_r, theory_t]
            for i, array in enumerate(results): 
                # store in info table
                table[counter,3+i] = array
    
    
            with open('data/fulldata-{}.txt'.format(counter), 'w') as file:
                # Write four lines to the file
                file.write("{} {} {}\n".format(nwname, n, m))
                file.write(' '.join(map(str, nw_r))+"\n")
                file.write(' '.join(map(str, nw_t))+"\n")
                file.write(' '.join(map(str, theory_r))+"\n")
                file.write(' '.join(map(str, theory_t))+"\n")

            print(' in {} s'.format(time.time()-t0))
        
        counter+=1

    if min_counter==0:
        # remove empty rows from table
        table2 = table[:counter]
    
        # convert to data frame and name its columns
        df = pd.DataFrame(table2)
        df.columns = ["network", "nodes", "edges", "real rand rLCC", "real attack rLCC",
                      "fin theory rand rLCC", "fin theory attack rLCC"]
        #print(info_table)
        return df
    else:
        return 0


nwks_notdone = []

with open(r'fullData.csv', 'a', newline='') as csvfile:
    for i in range(len(nwks_list2)):
        file = nwks_list2.iloc[i][1]
        file = file[60:]
        list_names = csvfile.sourceFileName.values()
        if file not in list_names:
            nwks_notdone.append(file)

#newdata = fullDataTable(nwks_notdone, num_tries=100, max_size=100, min_counter=0)
#newdata.to_csv("fullData2.csv")


# with open(r'fullData.csv', 'a', newline='') as csvfile:
#     for i in range(len(nwks_list2)):
#         file = nwks_list2.iloc[i][1]
#         file = file[60:]
#         list_names = csvfile.sourceFileName.values()
#         if file not in list_names:
#             nodes = nodecount_edge(file)[0]
#             edges = nodecount_edge(file)[1]
#         realrand = 
#             fieldnames = [file,nodes,edges,"real rand rLCC", "real attack rLCC",
#                             "fin theory rand rLCC", "fin theory attack rLCC"]
#             writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

#             writer.writerow({'This':'is', 'aNew':'Row'})

# #df.to_csv('log.csv', mode='a', index=False, header=False)