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



#print(finiteTheory.relSCurve(.0505, 100, attack=False, fdict=fvals, pdict=pvals, lcc_method_relS="pmult"))
#print(finiteTheory.relSCurve(.06756, 75, attack=False, fdict=fvals, pdict=pvals, lcc_method_relS="pmult"))
#print(finiteTheory.relSCurve(.06756, 75, attack=False, fdict=fvals, pdict=pvals, lcc_method_relS="alice"))
#
# def rel_LCC_table(np_list):
#     table = np.zeros((len(np_list), 6), dtype=object)
#     for i in range(len(np_list)):
#         table[i][0] = np_list[i][0]
#         table[i][1] = np_list[i][1]
#         table[i][2] = finiteTheory.relSCurve(np_list[i][1], np_list[i][0], attack=False, fdict=fvals, pdict=pvals, lcc_method_relS="pmult")
#         table[i][3] = finiteTheory.relSCurve(np_list[i][1], np_list[i][0], attack=True, fdict=fvals, pdict=pvals, lcc_method_relS="pmult")
#         table[i][4] = finiteTheory.relSCurve(np_list[i][1], np_list[i][0], attack=False, fdict=fvals, pdict=pvals,
#                                              lcc_method_relS="alice")
#         table[i][5] = finiteTheory.relSCurve(np_list[i][1], np_list[i][0], attack=True, fdict=fvals, pdict=pvals,
#                                              lcc_method_relS="alice")
#
#     df = pd.DataFrame(table)
#     df.columns = ["nodes", "prob", "rand pmult", "attack pmult", "rand alice",
#                   "attack alice"]
#     return df
#
# glitch_fix = rel_LCC_table([[75,.06756],[100,.0505]])
# glitch_fix.to_pickle("relLCCtable.p")




#
# fig = plot_graphs(numbers_of_nodes=[100], edge_probabilities=[.1],
#     graph_types=['ER'], remove_strategies=['random'],
#     performance='relative LCC', num_trials=10,
#     smooth_end=False, forbidden_values=[], fdict=fvals, lcc_method_main = "pmult", savefig='')
# fig.savefig("testfig.png")

def one_perc_thresh(threshold=.2, nodes=[10,20,30,40,50,60], removal = ["random"]):
    if removal == ["random"]:
        remove_bool = False
    elif removal == ["attack"]:
        remove_bool = True
    percthresh = threshold
    nodes_array = nodes
    prob_array = np.zeros(len(nodes_array))
    colors = ["red", "orange","yellow","green","purple","magenta","cyan"]

    for i in range(len(nodes_array)):
        prob_array[i] = 1/(percthresh*(nodes_array[i]-1))

    fig = plot_graphs(numbers_of_nodes=[nodes_array[0]], edge_probabilities=[prob_array[0]],
                      graph_types=['ER'], remove_strategies=removal,
                      performance='relative LCC', num_trials=100,
                      smooth_end=False, forbidden_values=[], fdict=fvals, lcc_method_main="pmult", savefig='',
                      simbool=True)

    for j in range(len(nodes_array)-1):
        sim_data = completeRCData(numbers_of_nodes=[nodes_array[j+1]],
                                  edge_probabilities=[prob_array[j+1]], num_trials=100,
                                  performance='relative LCC', graph_types=['ER'],
                                  remove_strategies=removal)
        data_array = np.array(sim_data[0][0][0][0])

        # exclude the first row, because it is the number of nodes
        data_array = data_array[1:]

        # this can prevent some sort of bug about invalid values
        for val in []:
            data_array[data_array == val] = np.nan

        # plot simulated data
        removed_fraction = np.arange(nodes_array[j+1]) / nodes_array[j+1]
        line_data = np.nanmean(data_array,axis=0)

        plt.plot(removed_fraction, line_data,
                     'o', label="n={} , p={}".format(nodes_array[j+1], prob_array[j+1]), color = colors[j])
        plt.plot(np.arange(nodes_array[j+1])/nodes_array[j+1],finiteTheory.relSCurve(prob_array[j+1],nodes_array[j+1],
                                attack=remove_bool, fdict=fvals,pdict=pvals,lcc_method_relS="pmult"),
                                label = "n: "+str(nodes_array[j+1]), color = colors[j])

    plt.legend()
    fig.savefig("testfig.png")

#one_perc_thresh(threshold=.2, nodes=[100], removal = ["random"])

def one_perc_thresh_table(threshold=.5, nodes=[10, 20, 30, 40, 50, 60], removal=["attack"]):
    one_perc_table = np.zeros((len(nodes), 4), dtype=object)

    if removal == ["random"]:
        remove_bool = False
    elif removal == ["attack"]:
        remove_bool = True
    percthresh = threshold
    nodes_array = nodes
    prob_array = np.zeros(len(nodes_array))
    colors = ["red", "orange", "yellow", "green", "purple", "magenta", "cyan"]

    for i in range(len(nodes_array)):
        prob_array[i] = 1 / (percthresh * (nodes_array[i] - 1))

    fig = plot_graphs(numbers_of_nodes=[nodes_array[0]], edge_probabilities=[prob_array[0]],
                      graph_types=['ER'], remove_strategies=removal,
                      performance='relative LCC', num_trials=100,
                      smooth_end=False, forbidden_values=[], fdict=fvals, lcc_method_main="pmult", savefig='',
                      simbool=True)

    for j in range(len(nodes_array)):
        sim_data = completeRCData(numbers_of_nodes=[nodes_array[j]],
                                  edge_probabilities=[prob_array[j]], num_trials=100,
                                  performance='relative LCC', graph_types=['ER'],
                                  remove_strategies=removal)
        data_array = np.array(sim_data[0][0][0][0])

        # exclude the first row, because it is the number of nodes
        data_array = data_array[1:]

        # this can prevent some sort of bug about invalid values
        for val in []:
            data_array[data_array == val] = np.nan

        # plot simulated data
        removed_fraction = np.arange(nodes_array[j]) / nodes_array[j]
        line_data = np.nanmean(data_array, axis=0)

        one_perc_table[j][0] = nodes_array[j]
        one_perc_table[j][1] = prob_array[j]
        one_perc_table[j][2] = line_data
        one_perc_table[j][3] = finiteTheory.relSCurve(prob_array[j], nodes_array[j],
                                        attack=remove_bool, fdict=fvals, pdict=pvals, lcc_method_relS="pmult")
        if j != 0:
            plt.plot(removed_fraction, line_data,
                     'o', label="n={} , p={}".format(nodes_array[j], prob_array[j]), color=colors[j])
            plt.plot(np.arange(nodes_array[j]) / nodes_array[j],
                     finiteTheory.relSCurve(prob_array[j], nodes_array[j],
                                            attack=remove_bool, fdict=fvals, pdict=pvals, lcc_method_relS="pmult"),
                     label="n: " + str(nodes_array[j]), color=colors[j])

    plt.legend()
    fig.savefig("testfig.png")
    df = pd.DataFrame(one_perc_table)
    df.columns = ["nodes", "prob", "simulated RLCC", "fin theory RLCC"]
    return df


# big_graph_r = one_perc_thresh_table(threshold=.2, nodes=[10, 15, 25, 50, 75, 100], removal=["random"])
# big_graph_r.to_pickle("percolation_random")
#
# big_graph_t = one_perc_thresh_table(threshold=.2, nodes=[10, 15, 25, 50, 75, 100], removal=["attack"])
# big_graph_t.to_pickle("percolation_attack")



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
        elif fnmatch(name, pattern2):
            nwks_list2.append(os.path.join(path, name))

# main?
#
# import time
# t0 = time.time()
# table_info(False,nwks_list2[:6])
# print(time.time()-t0)




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


# plot some together with similar sizes



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



# G_0 = nx.from_numpy_array(adj)
#                 G = G_0.copy()
#                 # nx.draw(G)
#                 # plt.show()
#
#                 p = len(edge_list) / scipy.special.comb(n, 2)
#                 remove_strat = [removal]
#
#                 x_array = np.arange(0, n) / n
#                 averaged_data = np.zeros(n)
#                 for j_2 in range(100):
#                     G = G_0.copy()
#                     # print(list(G.nodes()), "nodes")
#                     data_array = np.zeros(n, dtype=float)
#                     # print("averaged data", averaged_data)
#                     # print("data array", data_array)
#                     for i_2 in range(n):
#                         data_array[i_2] = len(max(nx.connected_components(G), key=len)) / (n - i_2)
#                         # find a node to remove
#                         if removal == "random":
#                             if G.number_of_nodes() != 0:
#                                 v = choice(list(G.nodes()))
#                                 G.remove_node(v)
#                                 # print(v)
#                         elif removal == "attack":
#                             if G.number_of_nodes() != 0:
#                                 v = sorted(G.degree, key=lambda x: x[1], reverse=True)[0][0]
#                                 G.remove_node(v)
#
#                     averaged_data += data_array
#                     # print("averaged data", averaged_data)
#                     # print("data array", data_array)
#                 print(x_array, "xarray")
#                 averaged_data /= 100
#                 plt.plot(x_array, averaged_data, label=file_name + "nodes: " + str(n))
#                 plt.legend()
#
#                 file.close()
#             fig.supxlabel("percent nodes removed")
#             fig.supylabel("Rel LCC")
#             if oneplot == True:
#                 fig.savefig(file_name[:-4] + ".png")
#
#             file.close()
#         if oneplot == False:
#             fig.savefig("realntwks.png")

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

# df = pd.read_csv("fullData.csv")
# print(type(df.iloc[0][3]))

# # same node average
#
# import ast
# df = pd.read_pickle("fullData.csv")
# aucftr = df["fin theory rand auc"].values
# node_list = df["nodes"].values
# tot_networks = len(df)
# max_netwk = node_list.max()
# aucftr_avg = np.zeros(max_netwk,dtype=object)
# for k1 in range(max_netwk):
#     aucftr_avg[k1] = np.zeros(k1)
# for k in range(tot_networks):
#     size = node_list[k]
#     print(aucftr[k])
#     newauc = aucftr[k]
#     newauc = newauc.strip('][').split(', ')
#     print(newauc)
#     newauc = np.asarray(newauc, dtype=float)
#     print(type(newauc))
#     print(newauc)
#     aucftr_avg[size] += newauc
#
# print(aucftr_avg)



# binning

# from ast import literal_eval
# fig = plt.figure(figsize=(8, 8))
# df = pd.read_csv("fullData.csv")
# # for i in range(len(df)):
# #     hist, bins = np.histogram(literal_eval(df.iloc[i][5]), bins=4) # fin theory random
# #     hist2, bins2 = np.histogram(literal_eval(df.iloc[i][6]), bins=4) # fin theory attack
# #     hist3, bins3 = np.histogram(np.linspace(0,1,literal_eval(df.iloc[i][1])), bins = 4) # percent nodes removed
# # plt.plot(bins3,bins,label = "random")
# # plt.plot(bins3,bins2, label = "attack")
# # fig.savefig("testfig.png")
#
# fig = plt.figure(figsize=(8, 8))
# df = pd.read_pickle("fullData.csv")
# bin_array1 = []
# bin_array2 = []
# bin_array3 = []
# bin_array4 = []
# nodes_array = np.zeros(len(df))
# for i in range(len(df)):
#     nodes_array[i] = float(df.iloc[i][1])
#
# hist, bins = np.histogram(nodes_array, bins=4)
# print("Bin edges", bins)
# for j in range(len(nodes_array)):
#     counter = 0
#     while nodes_array[j] > bins[counter]:
#         counter += 1
#     if counter == 0:
#         bin_array1.append(j)
#     elif counter == 1:
#         bin_array2.append(j)
#     elif counter == 2:
#         bin_array3.append(j)
#     else:
#         bin_array4.append(j)
# print(bin_array1)
# print(bin_array2)
# print(bin_array3)
# print(bin_array4)


df = pd.read_csv("AUC.csv")
x = np.zeros(len(df))
real_rauc = np.zeros(len(df))
real_tauc = np.zeros(len(df))
rauc = np.zeros(len(df))
tauc = np.zeros(len(df))
densities = np.zeros(len(df))
rlsqr = np.zeros(len(df))
tlsqr = np.zeros(len(df))

for i in range(len(df)):
    n = df.iloc[i][1]
    x[i] = n
    real_rauc[i] = df.iloc[i][3]
    real_tauc[i] = df.iloc[i][4]
    rauc[i] = df.iloc[i][5]
    tauc[i] = df.iloc[i][6]
    densities[i] = df.iloc[i][2] / (n*(n-1)/2)
    rlsqr[i] = (df.iloc[i][5] - df.iloc[i][3])**2
    tlsqr[i] = (df.iloc[i][6] - df.iloc[i][4])**2
    # if densities[i] > .9 and real_rauc[i] < .6:
    #     print("weird")
    #     p = df.iloc[i][2] / scipy.special.comb(n, 2)
    #     fig = plot_graphs(numbers_of_nodes=[n], edge_probabilities=[p],
    #         graph_types=['ER'], remove_strategies=['random'],
    #         performance='relative LCC', num_trials=10,
    #         smooth_end=False, forbidden_values=[], fdict=fvals, lcc_method_main = "pmult", savefig='')
    #     fig.savefig("testfig.png")

r_rbin_means,r_rbin_edges, r_rbinnumber = scipy.stats.binned_statistic(x, real_rauc, statistic='mean', bins=10, range=None)
r_tbin_means,r_tbin_edges, r_tbinnumber = scipy.stats.binned_statistic(x, real_tauc, statistic='mean', bins=10, range=None)

rbin_means,rbin_edges, rbinnumber = scipy.stats.binned_statistic(x, rauc, statistic='mean', bins=10, range=None)
tbin_means, tbin_edges, tbinnumber = scipy.stats.binned_statistic(x, tauc, statistic='mean', bins=10, range=None)
#binned by density
drbin_means, drbin_edges, drbinnumber = scipy.stats.binned_statistic(densities, rauc, statistic='mean', bins=10, range=None)
dtbin_means, dtbin_edges, dtbinnumber = scipy.stats.binned_statistic(densities, tauc, statistic='mean', bins=10, range=None)
#
# plt.plot(x, real_rauc,'o')
# plt.hlines(r_rbin_means, r_rbin_edges[:-1], r_rbin_edges[1:], colors='g', lw=2,
#            label='random removal auc')
#
# plt.plot(x, real_tauc,'o')
# plt.hlines(r_tbin_means, r_tbin_edges[:-1], r_tbin_edges[1:], colors='g', lw=2,
#            label='random removal auc')


# plt.plot(x, rauc,'o')
# plt.hlines(rbin_means, rbin_edges[:-1], rbin_edges[1:], colors='g', lw=2,
#            label='random removal auc')
#
# plt.plot(x, tauc,'o')
# plt.hlines(tbin_means, tbin_edges[:-1], tbin_edges[1:], colors='r', lw=2,
#            label='targeted removal auc')

# plt.xlabel("nodes")
# plt.ylabel("auc")

# plt.plot(densities, rauc,'o')
# plt.hlines(drbin_means, drbin_edges[:-1], drbin_edges[1:], colors='g', lw=2,
#            label='binned statistic of data')

# plt.plot(densities, tauc,'o')
# plt.hlines(dtbin_means, dtbin_edges[:-1], dtbin_edges[1:], colors='g', lw=2,
#            label='binned statistic of data')

#plt.xlabel("density")
# plt.xlabel("nodes in graph")
# plt.ylabel("auc")
# plt.legend()
# plt.show()

# least squares error graph
# max_node = int(max(x))
# lsqrx = np.arange(1,max_node+1)
# lsqryr = np.zeros(max_node)
# lsqryt = np.zeros(max_node)
# for j in range(len(x)):
#     i_l = df.iloc[j][1]-1
#     lsqryr[i_l] += rlsqr[j]
#     lsqryt[i_l] += tlsqr[j]

# plt.plot(lsqrx,lsqryt)
# plt.show()

#
# def bin_average(bin_array, attack=False):
#     average = 0
#     for i in bin_array:
#         if attack == False:
#             average += df.iloc[i][5]
#         else:
#             average += df.iloc[i][6]
#     average /= len(bin_array)
#     return average
#
# auc_plot = np.zeros(4)
# bins = [bin_array1, bin_array2, bin_array3, bin_array4]
# for ib in range(len(bins)):
#     auc_plot[ib] = bin_average(bins[ib])
# print(auc_plot)
#
# plt.plot()
#
#
# # hist, bins = np.histogram(np.linspace(0,1,literal_eval(df.iloc[i][1])), bins = 4) # percent nodes removed
# # hist, bins = np.histogram(literal_eval(df.iloc[i][5]), bins=4)  # fin theory random
# # hist2, bins2 = np.histogram(literal_eval(df.iloc[i][6]), bins=4)  # fin theory attack
# # plt.plot(bins3,bins,label = "random")
# # plt.plot(bins3,bins2, label = "attack")
# # fig.savefig("testfig.png")


def gradient(nodenumber, probsnumber):
    nodes = np.arange(1,nodenumber,1)
    probs = np.linspace(0,1,probsnumber+1)
    # frauc = np.zeros(len(nodes)*len(probs))
    # ftauc = np.zeros(len(nodes)*len(probs))
    # irauc = np.zeros(len(nodes)*len(probs))
    # itauc = np.zeros(len(nodes)*len(probs))
    srauc = np.zeros(len(nodes)*len(probs))
    # stauc = np.zeros(len(nodes)*len(probs))
    diff = np.zeros(len(nodes)*len(probs))
    grid_table = np.zeros((len(nodes)*len(probs),14),dtype=object)
    counter = 0
    heatmap_array = np.zeros((len(probs), len(nodes)), dtype=object)


    for i_n in range(len(nodes)):
        for i_p in range(len(probs)):
            grid_table[counter][0] = nodes[i_n]
            grid_table[counter][1] = probs[i_p]

            # rfin_curve = finiteTheory.relSCurve(probs[i_p],nodes[i_n],
            #                    attack=False, fdict=fvals, pdict=pvals, lcc_method_relS="pmult")
            # tfin_curve = finiteTheory.relSCurve(probs[i_p],nodes[i_n],
            #                    attack=True, fdict=fvals, pdict=pvals, lcc_method_relS="pmult")
            rinf_curve = infiniteTheory.relSCurve(nodes[i_n],probs[i_p],
                                 attack=False, smooth_end=False)
            # tinf_curve = infiniteTheory.relSCurve(nodes[i_n],probs[i_p],
            #                     attack=True, smooth_end=False)
            # rsim_data = completeRCData(numbers_of_nodes=[nodes[i_n]],
            #                            edge_probabilities=[probs[i_p]], num_trials=100,
            #                            performance='relative LCC', graph_types=['ER'],
            #                            remove_strategies=["random"])
            #
            # rdata_array = np.array(rsim_data[0][0][0][0])
            #
            # # exclude the first row, because it is the number of nodes
            # rdata_array = rdata_array[1:]
            #
            # # this can prevent some sort of bug about invalid values
            # for val in []:
            #     rdata_array[rdata_array == val] = np.nan
            #
            # rline_data = np.nanmean(rdata_array,axis=0)
            #
            # tsim_data = completeRCData(numbers_of_nodes=[nodes[i_n]],
            #                            edge_probabilities=[probs[i_p]], num_trials=100,
            #                            performance='relative LCC', graph_types=['ER'],
            #                            remove_strategies=["attack"])
            #
            # tdata_array = np.array(tsim_data[0][0][0][0])
            #
            # # exclude the first row, because it is the number of nodes
            # tdata_array = tdata_array[1:]
            #
            # # this can prevent some sort of bug about invalid values
            # for val in []:
            #     tdata_array[tdata_array == val] = np.nan
            #
            # tline_data = np.nanmean(tdata_array, axis=0)

            # grid_table[counter][2] = scipy.integrate.simpson(rfin_curve, dx=1 / nodes[i_n])
            # grid_table[counter][2] = ((rfin_curve - rline_data) ** 2).mean()
            # grid_table[counter][3] = rfin_curve
            # frauc[counter] = grid_table[counter][2]
            # grid_table[counter][4] = scipy.integrate.simpson(tfin_curve, dx=1 / nodes[i_n])
            # grid_table[counter][4] = ((tfin_curve - tline_data) ** 2).mean()
            # grid_table[counter][5] = tfin_curve
            # ftauc[counter] = grid_table[counter][4]
            grid_table[counter][6] = scipy.integrate.simpson(rinf_curve, dx=1 / nodes[i_n])
            # grid_table[counter][6] = ((rinf_curve - rline_data) ** 2).mean()
            # grid_table[counter][7] = rinf_curve
            # irauc[counter] = grid_table[counter][6]
            # grid_table[counter][8] = scipy.integrate.simpson(tinf_curve, dx=1 / nodes[i_n])
            # grid_table[counter][8] = ((tinf_curve - tline_data) ** 2).mean()
            # grid_table[counter][9] = tinf_curve
            # itauc[counter] = grid_table[counter][8]
            # grid_table[counter][10] = scipy.integrate.simpson(rline_data, dx=1 / nodes[i_n])
            # grid_table[counter][11] = rline_data
            # srauc[counter] = grid_table[counter][10]
            # grid_table[counter][12] = scipy.integrate.simpson(tline_data, dx=1 / nodes[i_n])
            # grid_table[counter][13] = tline_data
            # stauc[counter] = grid_table[counter][12]

            # heatmap_array[i2][i1] = scipy.integrate.simpson(finiteTheory.relSCurve(probs[i2],nodes[i1],
            #                     attack=False, fdict=fvals, pdict=pvals, lcc_method_relS="pmult"), dx=1 / nodes[i1])

            # heatmap_array[i2][i1] = scipy.integrate.simpson(infiniteTheory.relSCurve(nodes[i1],probs[i2],
            #                     attack=False, smooth_end=False))
            # heatmap_array[i2][i1] = scipy.integrate.simpson(finiteTheory.relSCurve(probs[i2],nodes[i1],
            #                      attack=False, fdict=fvals, pdict=pvals, lcc_method_relS="pmult"), dx=1 / nodes[i1]) - scipy.integrate.simpson(infiniteTheory.relSCurve(nodes[i1], probs[i2],
            #                                                                          attack=False, smooth_end=False))
            # heatmap_array[i_p][i_n] = scipy.integrate.simpson(rline_data, dx=1 / nodes[i_n])
            heatmap_array[i_p][i_n] = grid_table[counter][6]
            counter += 1
            print("counter",counter)
    heatmap_array = heatmap_array.tolist()

    # nodes have to be 1,1,1,...,2,2,2...etc
    # n= len(nodes)
    # p= len(probs)
    #
    # newn = np.zeros(n*p)
    # newp = np.zeros(n*p)
    # newn = np.repeat(nodes, p)
    # counter = 0
    # for k in range(n):
    #     for j in range(p):
    #         newp[counter] = probs[j]
    #         counter +=1

    # print(heatmap_array)
    # for i1 in range(len(newn)):
    #     heatmap_array[0][i1] = [newn[i1],newp[i1]]
    #     heatmap_array[1][i1] = frauc[i1]
    xnodes, yprobs = np.meshgrid(nodes,probs)
    print("x",xnodes)
    print("y", yprobs)

    print("heatmap",heatmap_array)
    # heatmap_array2 = [[0.0, 0.375, 0.37037037037037035, 0.35590277777777773],
    #                   [0.0, 0.4, 0.43674074074074076, 0.46548888888888895],
    #                   [0.0, 0.425, 0.5013333333333334, 0.566225],
    #                   [0.0, 0.45, 0.5623703703703704, 0.6502777777777777],
    #                   [0.0, 0.475, 0.6180740740740741, 0.711813888888889],
    #                   [0.0, 0.5, 0.6666666666666666, 0.75]]
    # probs2 = np.repeat(probs,n,axis=1)
    # newp = np.empty(n*p)
    # for l in probs2:
    #     newp += l
    plt.pcolormesh(xnodes, yprobs, heatmap_array)
    plt.xlabel("nodes")
    plt.ylabel("probability")
    plt.title("heatmap of AUC MSE")
    plt.colorbar()  # need a colorbar to show the intensity scale

    #plt.scatter(newn, newp , c=frauc, s=100, cmap="tab10")
    #plt.imshow(heatmap_array,"tab10")
    plt.show()
    df = pd.DataFrame(grid_table)
    return df


def gradient(nodenumber, probsnumber, removal, mse_type):
    nodes = np.arange(1, nodenumber + 1, 1)
    probs = np.linspace(0, 1, num=probsnumber + 1)  # probsnumber = 10 means gaps of .1
    grid_table = np.zeros((nodenumber, probsnumber + 1), dtype=object)
    heatmap_array = np.zeros((probsnumber + 1, nodenumber), dtype=object)

    if removal == "random":
        remove_bool = False
    elif removal == "attack":
        remove_bool = True

    for i_n in range(nodenumber):
        for i_p in range(probsnumber):
            sim_data = completeRCData(numbers_of_nodes=[nodes[i_n]],
                                      edge_probabilities=[probs[i_p]], num_trials=100,
                                      performance='relative LCC', graph_types=['ER'],
                                      remove_strategies=[removal])
            rdata_array = np.array(sim_data[0][0][0][0])
            rdata_array = rdata_array[1:]
            for val in []:
                rdata_array[rdata_array == val] = np.nan

            line_data = np.nanmean(rdata_array, axis=0)

            if mse_type == "simulations":
                heatmap_array[i_p][i_n] = scipy.integrate.simpson(line_data, dx=1 / (nodes[i_n] - 1))
                grid_table[i_n][i_p] = line_data
            elif mse_type == "finite":
                fin_curve = finiteTheory.relSCurve(probs[i_p], nodes[i_n],
                                                   attack=remove_bool, fdict=fvals, pdict=pvals,
                                                   lcc_method_relS="pmult")
                heatmap_array[i_p][i_n] = ((fin_curve - line_data) ** 2).mean()
                grid_table[i_n][i_p] = np.zeros(2,dtype=object)
                grid_table[i_n][i_p][0] = line_data
                grid_table[i_n][i_p][1] = fin_curve
            elif mse_type == "infinite":
                inf_curve = infiniteTheory.relSCurve(nodes[i_n], probs[i_p],
                                                     attack=remove_bool, smooth_end=False)
                heatmap_array[i_p][i_n] = ((inf_curve - line_data) ** 2).mean()
                grid_table[i_n][i_p] = np.zeros(2,dtype=object)
                grid_table[i_n][i_p][0] = line_data
                grid_table[i_n][i_p][1] = inf_curve
            elif mse_type == "difference":
                fin_curve = finiteTheory.relSCurve(probs[i_p], nodes[i_n],
                                                   attack=remove_bool, fdict=fvals, pdict=pvals,
                                                   lcc_method_relS="pmult")
                inf_curve = infiniteTheory.relSCurve(nodes[i_n], probs[i_p],
                                                     attack=remove_bool, smooth_end=False)
                heatmap_array[i_p][i_n] = ((inf_curve - line_data) ** 2).mean() - ((fin_curve - line_data) ** 2).mean()
                grid_table[i_n][i_p] = np.zeros(3,dtype=object)
                grid_table[i_n][i_p][0] = line_data
                grid_table[i_n][i_p][1] = fin_curve
                grid_table[i_n][i_p][2] = inf_curve

    heatmap_array = heatmap_array.tolist()

    xnodes, yprobs = np.meshgrid(nodes, probs)

    plt.pcolormesh(xnodes, yprobs, heatmap_array)
    plt.xlabel("nodes")
    plt.ylabel("probability")
    plt.title("heatmap of AUC MSE")
    plt.colorbar()  # need a colorbar to show the intensity scale
    plt.show()
    df = pd.DataFrame(grid_table)
    return df

#output = gradient(50,10,"random", "finite")
#output = gradient(5,5,"random", "finite")
#output.to_pickle("gradient 50n 10p fin")


# obj = pd.read_pickle(r'gradient 50n 10p fin')
# n = int(obj.iloc[-1][0])
# print(n)
# p = int(len(obj) / n)
# print(p)
# nodes = np.arange(1,n+1,1)
# probs = np.linspace(0,1,p+1)
# x,y = np.meshgrid(nodes,probs)
# heatmap_array = np.zeros((len(probs), len(nodes)), dtype=object)
# counter = 0
# for i in range(n):
#     for j in range(p):
#         #print(i*p+j-1)
#         print(obj.iloc[i*p+j-1][2])
#         heatmap_array[j][i] = obj.iloc[i*p+j-1][2]
# print(heatmap_array)
# heatmap_array = heatmap_array.tolist()
# plt.pcolormesh(x,y,heatmap_array,cmap = "Reds")
# print("done")
# plt.savefig("colormap.png")

def check_space(string):
    # counter
    count = 0

    # loop for search each index
    for i in range(0, len(string)):

        # Check each char
        # is blank or not
        if string[i] == " ":
            count += 1

    return count

def bayesian(theory = False, removal = "random", adj_list = ["taro.txt"], oneplot = False):
    if theory == True:
        fig = plot_graphs(numbers_of_nodes=[n], edge_probabilities=[p],
                          graph_types=['ER'], remove_strategies=[removal],
                          performance='relative LCC', num_trials=100,
                          smooth_end=False, forbidden_values=[], fdict=fvals, lcc_method_main="alice", savefig='')
    else:
        fig = plt.figure(figsize=(8, 8))
    for file_name in adj_list:
        print(str(file_name))
        file = open(file_name, "r")
        #content = file.readlines()
        content = (line.rstrip() for line in file)  # All lines including the blank ones
        content = list(line for line in content if line)

        if len(content) == 0:
            file.close()
        # identify type of file (biadjacency matrix, edge list)
            # edge list
        elif check_space(content[0])==1 and (len(content) == 1 or len(content) > 2):
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
                print("check1")

                for k in range(len(edge_list)):
                    adj[int(edge_list[k][0]), int(edge_list[k][1])] = 1
                    adj[int(edge_list[k][1]), int(edge_list[k][0])] = 1
                G_0 = nx.from_numpy_array(adj)
                p = len(edge_list) / scipy.special.comb(n, 2)
                degrees = list(G_0.degree())
                product = 1
                for i_d in range(len(degrees)):
                    d = degrees[i_d][1]
                    print("d", d)
                    product *= scipy.special.comb(n - 1, d) * (p ** d) * (1 - p) ** (n - 1 - d)
                    print("i_d", i_d)
                nx.draw(G_0)
                # plt.show()
                freq = nx.degree_histogram(G_0)
                print(freq)
                for f in freq:
                    product /= math.factorial(f)
                product = product * math.factorial(n)
                return product

            #biadjacency matrix
        elif len(content[0]) > 4 or (len(content[0]) == 4 and len(content) == 2):
            print("biadj file")
            if nodecount_bi(file_name) > 100:
                file.close()

            else:
                print("else")
                edge_list = np.empty(len(content), dtype=object)
                # edge1 = content[0].strip()
                # edge1 = edge1.split("\t")
                k = len(content[0].strip().split("\t"))
                n = len(content) + k
                adj1 = np.zeros((len(content), k))
                adj = np.zeros((n, n))
                print("adj beg", adj)

                for i in range(len(content)):
                    edge = content[i].strip()
                    edge = edge.split("\t")
                    print("edge")
                    print(edge)

                    for j in range(len(edge)):
                        e = int(float(edge[j]))
                        if e >= 1:
                            adj1[i, j] = 1
                            adj[i, j + len(content)] = 1
                            adj[j + len(content), i] = 1
                        else:
                            adj1[i, j] = 0
                            
                print("adj", adj)
                print("check2")
                print(adj)
                G_0 = nx.from_numpy_array(adj)
                p = len(edge_list) / scipy.special.comb(n, 2)
                degrees = list(G_0.degree())
                product = 1
                for i_d in range(len(degrees)):
                    d = degrees[i_d][1]
                    print("d",d)
                    product *= scipy.special.comb(n-1,d)*(p**d)*(1-p)**(n-1-d)
                    print("i_d",i_d)
                nx.draw(G_0)
                #plt.show()
                freq = nx.degree_histogram(G_0)
                print(freq)
                for f in freq:
                    product /= math.factorial(f)
                product = product * math.factorial(n)
                return product

#print(bayesian(theory = False, removal = "random", adj_list = ["AmadMyJn12-20.adj"], oneplot = False))
bayesian_array = np.zeros(len(nwks_list2))
counter=0
for nwk in nwks_list2:
    print(nwk)
    bayesian_array[counter] = bayesian(theory=False, removal = "random", adj_list = [nwk], oneplot = False)
bayesian = pd.DataFrame(bayesian_array)
#bayesian.to_pickle("bayesian array")
median = np.median(bayesian_array)
small = []
big = []
for i in range(len(nwks_list2)):
    if bayesian_array[i] >= median:
        small.append(nwks_list2[i])
    else:
        big.append(nwks_list2[i])
print(small)
small_df = pd.DataFrame(small)
small_df.to_pickle("small array")
print(big)
big_df = pd.DataFrame(big)
big_df.to_pickle("big array")

def std_bins(nodes = [10,15,20,25], probs = np.linspace(0,1,11), removal = "random", trials = 100):
    n = len(nodes)
    p = len(probs)
    rlcc_table = np.zeros((n, p), dtype=object)

    if removal == "random":
        remove_bool = False
    elif removal == "attack":
        remove_bool = True

    fig = plt.figure(figsize=(8, 8))

    for i_p in range(p):
        auc_theory = np.zeros(n)
        auc_simy = np.zeros(n)
        std_table = np.zeros(n)

        for i_n in range(n):
            auc_theory[i_n] = scipy.integrate.simpson(finiteTheory.relSCurve(probs[i_p], nodes[i_n],
                                   attack=remove_bool, fdict=fvals, pdict=pvals,
                                   lcc_method_relS="pmult"), dx=1/(nodes[i_n]-1))
            rlcc_table[i_n][i_p] = np.zeros(trials, dtype=object)

            auc_sim = np.zeros(trials)
            for i_s in range(trials):
                # rlcc_table[i_n][i_p][i_s] = np.zeros(n)
                sim_data = completeRCData(numbers_of_nodes=[nodes[i_n]],
                                          edge_probabilities=[probs[i_p]], num_trials=1,
                                          performance='relative LCC', graph_types=['ER'],
                                          remove_strategies=[removal])
                rdata_array = np.array(sim_data[0][0][0][0])
                rdata_array = rdata_array[1:]
                for val in []:
                    rdata_array[rdata_array == val] = np.nan

                line_data = np.nanmean(rdata_array, axis=0)
                rlcc_table[i_n][i_p][i_s] = line_data
                auc_sim[i_s] = scipy.integrate.simpson(line_data, dx=1/(nodes[i_n]-1))

            auc_simy[i_n] = np.nanmean(auc_sim)
            std = np.std(auc_sim)
            std_table[i_n] = std

            # avg = np.reshape(rlcc_table[i_n][i_p], (trials, nodes[i_n]))
            # avg = np.nanmean(avg, axis=0)
        plt.plot(nodes,auc_theory,label = str(probs[i_p]))
        plt.errorbar(x=nodes, y=auc_simy, yerr = 2*std_table)
    plt.xlabel("nodes")
    plt.ylabel("AUC")
    plt.show()
    df = pd.DataFrame(rlcc_table)
    return df

# std_bins([10,15,20,25],probs=[.1,.125,.15,.175,.2],removal = "random", trials=100)
#std_bins([10,12,15,20,25,35,50],probs=[.1],removal = "random", trials=100)
std_output = std_bins([50],probs=[.05,.08,.1],removal = "random", trials=100)
std_output.to_pickle("std n=50,p=.05 .08 .1")
