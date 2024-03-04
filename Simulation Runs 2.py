import sys, pickle
sys.path.insert(0, "libs")

import os, pickle, csv # import packages for file I/O
import time # package to help keep track of calculation time

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import scipy
import scipy.stats as sst
from scipy.special import comb
from scipy.signal import argrelextrema
from random import choice

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


#fig, (ax1, ax2) = plt.subplots(1, 2)
# ax1 = plot_graphs(numbers_of_nodes=[20], edge_probabilities=[.1,.2,.5],
#     graph_types=['ER'], remove_strategies=['attack'],
#     performance='relative LCC', num_trials=10,
#     smooth_end=False, forbidden_values=[], fdict=fvals, pdict=pvals, savefig='')
# ax2 = plot_graphs(numbers_of_nodes=[10,20,30], edge_probabilities=[.1],
#     graph_types=['ER'], remove_strategies=['attack'],
#     performance='relative LCC', num_trials=10,
#     smooth_end=False, forbidden_values=[], fdict=fvals, pdict=pvals, savefig='')
#
fig = plot_graphs(numbers_of_nodes=[20], edge_probabilities=[.1],
    graph_types=['ER'], remove_strategies=['attack'],
    performance='relative LCC', num_trials=100,
    smooth_end=False, forbidden_values=[], fdict=fvals, lcc_method_main = "pmult", savefig='')
fig.savefig("testfig.png")


def edge_files(removal = "random", theory = False, nwks_list = ["AmadMyJn12-51.adj"]):
    for file_name in nwks_list:
        print(file_name)
        file = open(file_name, "r")
        content=file.readlines()
        node_list=[]
        edge_list=np.empty(len(content),dtype=object)
        for i in range(len(content)):
            edge = content[i].strip()
            edge = edge.split(" ")
            edge_list[i] = np.zeros(2)
            edge_list[i][0] = int(edge[0])
            edge_list[i][1] = int(edge[1])
            for j in range(2):
                node_list.append(int(edge[j]))
        n = max(node_list)+1
        adj=np.zeros((n,n))

        for k in range(len(edge_list)):
            adj[int(edge_list[k][0]),int(edge_list[k][1])] = 1
            adj[int(edge_list[k][1]),int(edge_list[k][0])] = 1

        G_0 = nx.from_numpy_array(adj)
        G = G_0.copy()
        #nx.draw(G)
        #plt.show()

        p = len(edge_list)/scipy.special.comb(n,2)
        remove_strat = [removal]
        if theory == True:
            fig = plot_graphs(numbers_of_nodes=[n], edge_probabilities=[p],
                graph_types=['ER'], remove_strategies=[removal],
                performance='relative LCC', num_trials=100,
                smooth_end=False, forbidden_values=[], fdict=fvals, lcc_method_main = "alice", savefig='')
        else:
            fig = plt.figure(1,2, figsize=(8, 8))

        fig.savefig(file_name[:-4] + ".png")

        x_array = np.arange(0,n)/n
        averaged_data = np.zeros(n)
        for j_2 in range(100):
            G = G_0.copy()
            #print(list(G.nodes()), "nodes")
            data_array = np.zeros(n, dtype=float)
            # print("averaged data", averaged_data)
            # print("data array", data_array)
            for i_2 in range(n):
                data_array[i_2] = len(max(nx.connected_components(G), key=len)) / (n-i_2)
                # find a node to remove
                if removal == "random":
                    if G.number_of_nodes() != 0:
                        v = choice(list(G.nodes()))
                        G.remove_node(v)
                        #print(v)
                elif removal == "attack":
                    if G.number_of_nodes() != 0:
                        v = sorted(G.degree, key=lambda x: x[1], reverse=True)[0][0]
                        G.remove_node(v)

            averaged_data+=data_array
            # print("averaged data", averaged_data)
            # print("data array", data_array)
        print(x_array,"xarray")
        averaged_data /= 100
        plt.plot(x_array,averaged_data,label="real")
        plt.legend()
        fig.savefig(file_name[:-4]+".png")

        file.close()

#edge_files()
#edge_files("attack",False,["AmadMyJn12-51.adj", "AmadMyJn12-20.adj", "taro.txt",
    #"AmadMyJn12-51.adj", "10_19.adj", "BHGB10-20.adj", "bordeaux_ferry.adj"])


def edge_files1graph(removal="random", theory=False, nwks_list=["AmadMyJn12-51.adj"]):
    if theory == True:
        fig = plot_graphs(numbers_of_nodes=[n], edge_probabilities=[p],
                          graph_types=['ER'], remove_strategies=[removal],
                          performance='relative LCC', num_trials=100,
                          smooth_end=False, forbidden_values=[], fdict=fvals, lcc_method_main="alice", savefig='')
    else:
        fig = plt.figure(figsize=(8, 8))

    for file_name in nwks_list:
        print(file_name)
        file = open(file_name, "r")
        content = file.readlines()
        node_list = []
        edge_list = np.empty(len(content), dtype=object)
        for i in range(len(content)):
            edge = content[i].strip()
            edge = edge.split(" ")
            edge_list[i] = np.zeros(2)
            edge_list[i][0] = int(edge[0])
            edge_list[i][1] = int(edge[1])
            for j in range(2):
                node_list.append(int(edge[j]))
        n = max(node_list) + 1
        adj = np.zeros((n, n))

        for k in range(len(edge_list)):
            adj[int(edge_list[k][0]), int(edge_list[k][1])] = 1
            adj[int(edge_list[k][1]), int(edge_list[k][0])] = 1

        G_0 = nx.from_numpy_array(adj)
        G = G_0.copy()
        # nx.draw(G)
        # plt.show()

        p = len(edge_list) / scipy.special.comb(n, 2)
        remove_strat = [removal]

        x_array = np.arange(0, n) / n
        averaged_data = np.zeros(n)
        for j_2 in range(100):
            G = G_0.copy()
            # print(list(G.nodes()), "nodes")
            data_array = np.zeros(n, dtype=float)
            # print("averaged data", averaged_data)
            # print("data array", data_array)
            for i_2 in range(n):
                data_array[i_2] = len(max(nx.connected_components(G), key=len)) / (n - i_2)
                # find a node to remove
                if removal == "random":
                    if G.number_of_nodes() != 0:
                        v = choice(list(G.nodes()))
                        G.remove_node(v)
                        # print(v)
                elif removal == "attack":
                    if G.number_of_nodes() != 0:
                        v = sorted(G.degree, key=lambda x: x[1], reverse=True)[0][0]
                        G.remove_node(v)

            averaged_data += data_array
            # print("averaged data", averaged_data)
            # print("data array", data_array)
        print(x_array, "xarray")
        averaged_data /= 100
        plt.plot(x_array, averaged_data, label= file_name+"nodes: "+str(n))
        plt.legend()

        file.close()
    fig.supxlabel("percent nodes removed")
    fig.supylabel("Rel LCC")
    fig.savefig("realnwks.png")

edge_files1graph("attack",False,["AmadMyJn12-51.adj", "AmadMyJn12-20.adj", "taro.txt",
    "AmadMyJn12-51.adj", "10_19.adj", "BHGB10-20.adj", "bordeaux_ferry.adj"])


# for adj matrix files
def biadj_files(removal = "random", adj_list = ["Davidson & Fisher (1991) Plant-Ant.txt"]):
    for file_name in adj_list:
        print(file_name)
        file = open(file_name, "r")
        content=file.readlines()
        edge_list=np.empty(len(content),dtype=object)
        #edge1 = content[0].strip()
        #edge1 = edge1.split("\t")
        k = len(content[0].strip().split("\t"))
        n = len(content) + k
        adj = np.zeros((len(content),k))
        big_adj = np.zeros((n,n))

        for i in range(len(content)):
            edge = content[i].strip()
            edge = edge.split("\t")

            for j in range(len(edge)):
                e = int(float(edge[j]))
                if e >= 1:
                    adj[i,j] = 1
                    big_adj[i,j+len(content)] = 1
                    big_adj[j+len(content),i] = 1
                else:
                    adj[i,j] = 0
        G_0 = nx.from_numpy_array(big_adj)
        #G_0 = nx.algorithms.bipartite.matrix.from_biadjacency_matrix(adj, create_using=None)
        G = G_0.copy()
        nx.draw(G)
        plt.show()

        p = len(edge_list) / scipy.special.comb(n, 2)

        fig = plot_graphs(numbers_of_nodes=[n], edge_probabilities=[p],
                          graph_types=['ER'], remove_strategies=["random"],
                          performance='relative LCC', num_trials=100,
                          smooth_end=False, forbidden_values=[], fdict=fvals, lcc_method_main="alice", savefig='')
        fig.savefig(file_name[:-4]+".png")

        x_array = np.arange(0, n) / n
        averaged_data = np.zeros(n)
        for j_2 in range(100):
            G = G_0.copy()
            # print(list(G.nodes()), "nodes")
            data_array = np.zeros(n, dtype=float)
            # print("averaged data", averaged_data)
            # print("data array", data_array)
            for i_2 in range(n):
                print(G.number_of_nodes(), "g size before")
                data_array[i_2] = len(max(nx.connected_components(G), key=len)) / (n - i_2)
                # find a node to remove
                if removal == "random":
                    if G.number_of_nodes() != 0:
                        v = choice(list(G.nodes()))
                        G.remove_node(v)
                        # print(v)
                elif removal == "attack":
                    if G.number_of_nodes() != 0:
                        v = sorted(G.degree, key=lambda x: x[1], reverse=True)[0][0]
                        G.remove_node(v)
            averaged_data += data_array
            # print("averaged data", averaged_data)
            # print("data array", data_array)
        print(x_array, "xarray")
        averaged_data /= 100
        print(averaged_data, "y")
        plt.plot(x_array, averaged_data, label="real")
        plt.legend()
        fig.savefig(file_name[:-4]+".png")

        file.close()

#biadj_files("attack")