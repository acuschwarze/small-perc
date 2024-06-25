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

def one_perc_thresh(threshold=.5, nodes=[10,20,30,40,50,60], removal = ["attack"]):
    if removal == ["random"]:
        remove_bool = False
    elif removal == ["attack"]:
        remove_bool = True
    percthresh = threshold
    nodes_array = nodes
    prob_array = np.zeros(len(nodes_array))
    for i in range(len(nodes_array)):
        prob_array[i] = 1/(percthresh*(nodes_array[i]-1))
    print(prob_array)
    fig = plot_graphs(numbers_of_nodes=[nodes_array[0]], edge_probabilities=[prob_array[0]],
                      graph_types=['ER'], remove_strategies=removal,
                      performance='relative LCC', num_trials=10,
                      smooth_end=False, forbidden_values=[], fdict=fvals, lcc_method_main="pmult", savefig='',
                      simbool=True)
    for j in range(len(nodes_array)):
        plt.plot(np.arange(nodes_array[j])/nodes_array[j],finiteTheory.relSCurve(prob_array[j],nodes_array[j],
                                attack=remove_bool, fdict=fvals,pdict=pvals,lcc_method_relS="pmult"), label = "n: "+str(nodes_array[j]))
    plt.legend()
    fig.savefig("testfig.png")

#one_perc_thresh(.2 ,[10,20,30,40,50,60])



def table_info(theory=False, nwks_list=["dolphin.adj"]):

    fig = plt.figure(figsize=(8, 8))
    info_table = np.zeros((len(nwks_list),7),dtype=object)
    attackstat = False
    for remove_method in ["random", "attack"]:
        if remove_method == "random":
            attackstat = False
        else:
            attackstat = True
        counter = 0
        for file_name in nwks_list:
            info_table[counter,0] = str(file_name)
            #print(file_name)
            file = open(file_name, "r")
            content = file.readlines()
            if len(content) == 0:
                file.close()
            else:
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
                if 0 in node_list:
                    n = max(node_list) + 1
                else:
                    n = max(node_list)
                if n > 100:
                    file.close()
                else:
                    adj = np.zeros((n, n))

                info_table[counter, 1] = n
                info_table[counter, 2] = len(edge_list)

                for k in range(len(edge_list)):
                    adj[int(edge_list[k][0]), int(edge_list[k][1])] = 1
                    adj[int(edge_list[k][1]), int(edge_list[k][0])] = 1

                G_0 = nx.from_numpy_array(adj)
                G = G_0.copy()
                # nx.draw(G)
                # plt.show()

                p = len(edge_list) / scipy.special.comb(n, 2)

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
                        if remove_method == "random":
                            if G.number_of_nodes() != 0:
                                v = choice(list(G.nodes()))
                                G.remove_node(v)
                                # print(v)
                        elif remove_method == "attack":
                            if G.number_of_nodes() != 0:
                                v = sorted(G.degree, key=lambda x: x[1], reverse=True)[0][0]
                                G.remove_node(v)

                    averaged_data += data_array
                    # print("averaged data", averaged_data)
                    # print("data array", data_array)
                #print(x_array, "xarray")
                averaged_data /= 100
                area_real = scipy.integrate.simpson(averaged_data, dx=1/n)
                #print(attackstat)
                area_sim = scipy.integrate.simpson(relSCurve(p,n,attack = attackstat, reverse=True, fdict={}, pdict={}, lcc_method_relS="pmult"), dx=1/n)

                if remove_method == "random":
                    info_table[counter,3] = area_real
                    info_table[counter,5] = area_sim
                if remove_method == "attack":
                    info_table[counter,4] = area_real
                    info_table[counter,6] = area_sim


                plt.plot(x_array, averaged_data, label= file_name+"nodes: "+str(n))
                if theory == True:
                    plt.plot(x_array, infiniteTheory.relSCurve(n, p,
                            attack=attackstat, smooth_end=False), label = "inf theory" + str(attackstat))
                plt.legend()

                counter+=1
                file.close()

    fig.supxlabel("percent nodes removed")
    fig.supylabel("Rel LCC")
    fig.savefig("realnwks.png")
    df = pd.DataFrame(info_table)
    df.columns = ["network", "nodes", "edges", "real rand auc", "real attack auc",
                  "fin theory rand auc", "fin theory attack auc"]
    return df


#table_info(True, ["AmadMyJn12-20.adj"])

#print(table_info(False, ["AmadMyJn12-20.adj"]))
# print("done")
from fnmatch import fnmatch

root = r'C:\Users\jj\Downloads\GitHub\small-perc\nwks small perc'
pattern = "*.adj"
pattern2 = "*.arc"
nwks_list2 = []

for path, subdirs, files in os.walk(root):
    for name in files:
        if fnmatch(name, pattern):
            # print(os.path.join(path, name))
            nwks_list2.append(os.path.join(path, name))
        # elif fnmatch(name, pattern2):
        #     nwks_list2.append(os.path.join(path, name))

# main?

# import time
# t0 = time.time()
# table_info(False,nwks_list2[:6])
# print(time.time()-t0)




def nodecount_edge(file_name = ""):
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
    if 0 in node_list:
        n = max(node_list) + 1
    else:
        n = max(node_list)
    return n


def nodecount_bi(file_name = ""):
    file = open(file_name, "r")
    content = file.readlines()
    n = len(content)
    if len(content[0]) > n:
        n = len(content[0])
    return n


def mega_file_reader(theory = False, removal = "random", adj_list = ["taro.txt"], oneplot = False):
    if theory == True:
        fig = plot_graphs(numbers_of_nodes=[n], edge_probabilities=[p],
                          graph_types=['ER'], remove_strategies=[removal],
                          performance='relative LCC', num_trials=100,
                          smooth_end=False, forbidden_values=[], fdict=fvals, lcc_method_main="alice", savefig='')
    else:
        fig = plt.figure(figsize=(8, 8))
    for file_name in adj_list:
        print(file_name)
        file = open(file_name, "r")
        content=file.readlines()
        print("linecount")
        print(len(content))
        print(len(content[0]))
        print(content[0])
        if len(content) == 0:
            file.close()
        # identify type of file (biadjacency matrix, edge list)
            # edge list
        elif len(content[0]) == 4 and (len(content) == 1 or len(content) > 2):
            print('edge file')
            if nodecount_edge(file_name) > 100:
                file.close()
            else:
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
                plt.plot(x_array, averaged_data, label=file_name + "nodes: " + str(n))
                plt.legend()

                file.close()
            fig.supxlabel("percent nodes removed")
            fig.supylabel("Rel LCC")
            if oneplot == True:
                fig.savefig(file_name[:-4] + ".png")

            file.close()
        if oneplot == False:
            fig.savefig("realntwks.png")

            #biadjacency matrix
        elif len(content[0]) > 4 or (len(content[0]) == 4 and len(content) == 2):
            print("biadj file")
            if nodecount_bi(file_name) > 100:
                file.close()
            else:
                edge_list = np.empty(len(content), dtype=object)
                # edge1 = content[0].strip()
                # edge1 = edge1.split("\t")
                k = len(content[0].strip().split("\t"))
                n = len(content) + k
                adj = np.zeros((len(content), k))
                big_adj = np.zeros((n, n))

                for i in range(len(content)):
                    edge = content[i].strip()
                    edge = edge.split("\t")
                    print("edge")
                    print(edge)

                    for j in range(len(edge)):
                        e = int(float(edge[j]))
                        if e >= 1:
                            adj[i, j] = 1
                            big_adj[i, j + len(content)] = 1
                            big_adj[j + len(content), i] = 1
                        else:
                            adj[i, j] = 0
                G_0 = nx.from_numpy_array(big_adj)
                # G_0 = nx.algorithms.bipartite.matrix.from_biadjacency_matrix(adj, create_using=None)
                G = G_0.copy()
                nx.draw(G)
                plt.show()

                p = len(edge_list) / scipy.special.comb(n, 2)

                fig = plot_graphs(numbers_of_nodes=[n], edge_probabilities=[p],
                                  graph_types=['ER'], remove_strategies=["random"],
                                  performance='relative LCC', num_trials=100,
                                  smooth_end=False, forbidden_values=[], fdict=fvals, lcc_method_main="alice",
                                  savefig='')
                fig.savefig(file_name[:-4] + ".png")

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
                if oneplot == True:
                    fig.savefig(file_name[:-4] + ".png")

                file.close()
            if oneplot == False:
                fig.savefig("realntwks.png")

#mega_file_reader(False, "random", ["taro.txt", "Davidson & Fisher (1991) Plant-Ant.txt"])



# fig, (ax1, ax2) = plt.subplots(1, 2)
# ax1 = plot_graphs(numbers_of_nodes=[20], edge_probabilities=[.1,.2,.5],
#     graph_types=['ER'], remove_strategies=['attack'],
#     performance='relative LCC', num_trials=10,
#     smooth_end=False, forbidden_values=[], fdict=fvals, pdict=pvals, savefig='')
# ax2 = plot_graphs(numbers_of_nodes=[10,20,30], edge_probabilities=[.1],
#     graph_types=['ER'], remove_strategies=['attack'],
#     performance='relative LCC', num_trials=10,
#     smooth_end=False, forbidden_values=[], fdict=fvals, pdict=pvals, savefig='')

# fig = plot_graphs(numbers_of_nodes=[20], edge_probabilities=[.1],
#     graph_types=['ER'], remove_strategies=['attack'],
#     performance='relative LCC', num_trials=100,
#     smooth_end=False, forbidden_values=[], fdict=fvals, lcc_method_main = "pmult", savefig='')
# fig.savefig("testfig.png")




# same node average

import ast
df = pd.read_csv("fullData.csv")
aucftr = df["fin theory rand auc"].values
node_list = df["nodes"].values
tot_networks = len(df)
max_netwk = node_list.max()
aucftr_avg = np.zeros(max_netwk,dtype=object)
for k1 in range(max_netwk):
    aucftr_avg[k1] = np.zeros(k1)
for k in range(tot_networks):
    size = node_list[k]
    print(aucftr[k])
    newauc = aucftr[k]
    newauc = newauc.strip('][').split(', ')
    print(newauc)
    newauc = np.asarray(newauc, dtype=float)
    print(type(newauc))
    print(newauc)
    aucftr_avg[size] += newauc

print(aucftr_avg)



# binning

from ast import literal_eval
# fig = plt.figure(figsize=(8, 8))
# df = pd.read_csv("fullData.csv")
# for i in range(len(df)):
#     hist, bins = np.histogram(literal_eval(df.iloc[i][5]), bins=4) # fin theory random
#     hist2, bins2 = np.histogram(literal_eval(df.iloc[i][6]), bins=4) # fin theory attack
#     hist3, bins3 = np.histogram(np.linspace(0,1,literal_eval(df.iloc[i][1])), bins = 4) # percent nodes removed
# plt.plot(bins3,bins,label = "random")
# plt.plot(bins3,bins2, label = "attack")
# fig.savefig("testfig.png")

fig = plt.figure(figsize=(8, 8))
df = pd.read_csv("fullData.csv")
bin_array1 = []
bin_array2 = []
bin_array3 = []
bin_array4 = []
nodes_array = np.zeros(len(df))
for i in range(len(df)):
    nodes_array[i] = float(df.iloc[i][1])

hist, bins = np.histogram(nodes_array, bins=4)
print("Bin edges", bins)
for j in range(len(nodes_array)):
    counter = 0
    while nodes_array[j] > bins[counter]:
        counter += 1
    if counter == 0:
        bin_array1.append(j)
    elif counter == 1:
        bin_array2.append(j)
    elif counter == 2:
        bin_array3.append(j)
    else:
        bin_array4.append(j)
print(bin_array1)
print(bin_array2)
print(bin_array3)
print(bin_array4)

# hist, bins = np.histogram(np.linspace(0,1,literal_eval(df.iloc[i][1])), bins = 4) # percent nodes removed
# hist, bins = np.histogram(literal_eval(df.iloc[i][5]), bins=4)  # fin theory random
# hist2, bins2 = np.histogram(literal_eval(df.iloc[i][6]), bins=4)  # fin theory attack
# plt.plot(bins3,bins,label = "random")
# plt.plot(bins3,bins2, label = "attack")
# fig.savefig("testfig.png")
