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

# def deg_diff(n, p):
#     nvals = np.zeros(n)
#     pvals = np.zeros(n)
#     for i in range(n+1,0,-1):
#         nvals[i-1] = i
#         max = expectedMaxDegree(n, p)
#         mean = 2*scipy.
#         pvals[i-1] = diff



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
fig = plot_graphs(numbers_of_nodes=[34], edge_probabilities=[.1],
    graph_types=['ER'], remove_strategies=['random'],
    performance='relative LCC', num_trials=100,
    smooth_end=False, forbidden_values=[], fdict=fvals, lcc_method_main = "pmult", savefig='')
fig.savefig("testfig.png")

#print(calcA(.05,1,2))
#print(calcC(.05,2))

# print(calcA(.05,2,2))
# print(calculate_P(.05,0,2))
# print(calculate_P(.05,1,2))
# print(calculate_P(.05,2,2))
# print(calculate_P(.05,2,0))
#
# for i in range(3):
#     print("A"+str(i))
#     print(calcA(.05,i,2))
#     print("B"+str(i))
#     print(calcB(.05,i,2))

#calcC(.05,3)

#c_graph([20,15,25],[.05,.08, .1],to_vary="n")
#c_graph([21,15,18],[.05,.08,.1],to_vary="p")

def bump(n,i,p,para = "f"):
    # AS: This seems to be a function that compares simulated results and theory for f, kmax, and P?
    fig, (ax1, ax2) = plt.subplots(1, 2)
    if para == "f":
        for ip in range(len(p)):
            x_1 = np.zeros(i)
            y_1 = np.zeros(i)
            simx = np.zeros(i)
            simy = np.zeros(i)
            for ii1 in range(1,i):
                sizei = 0
                for r in range(1000):
                    G = nx.erdos_renyi_graph(n, p[ip])
                    if nx.is_connected(G.subgraph(range(ii1))):
                        sizei+=1
                simy[ii1] = sizei / 1000
                # take a graph, pick i nodes, see if they're connected
            for ii in range(i):
                x_1[ii] = ii
                y_1[ii] = raw_f(p[ip],ii,n)
                simx[ii] = ii

            ax1.plot(x_1,y_1,label = p[ip])
            ax1.set_xlabel('i')
            ax1.set_ylabel("raw f")
            ax2.plot(simx,simy, label = p[ip])
            ax2.set_xlabel('i')
            ax2.set_ylabel("simulated f")
    if para == "kmax":
        x_1 = np.zeros(i)
        y_1 = np.zeros(i)
        simx = np.zeros(i)
        simy = np.zeros(i)
        G = nx.erdos_renyi_graph(n, p)
        cc = [len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]
        for ik in range(1,n+1):
            x_1[ik-1] = (n - ik) / n
            y_1[ik-1] = expectedMaxDegree(ik, p)
            simx[ik-1] = (n - ik) / n
            gmax = len(max(nx.connected_components(G), key=len))
            simy[ik - 1] = gmax
            v = sorted(G.degree, key=lambda x: x[1], reverse=True)[0][0]
            G.remove_node(v)
            # for r in range(1000):
            #     G = nx.erdos_renyi_graph(ik,p)
            #     gmax = len(max(nx.connected_components(G), key=len))
            #     maxsum += gmax
            # simy[ik-1] = maxsum/1000
        simx = np.flip(simx)
        #simy = np.flip(simy)
        ax1.plot(x_1, y_1)
        ax1.set_xlabel('fraction n removed')
        ax1.set_ylabel("kmax")
        ax2.plot(simx, simy)
        ax2.set_xlabel('fraction n removed')
        ax2.set_ylabel("simulated kmax")

    if para == "P":
        for ip in range(len(p)):
            x_1 = np.zeros(i+1)
            y_1 = np.zeros(i+1)
            simx = np.zeros(i+1)
            simy = np.zeros(i+1)
            for ii in range(i+1):
                x_1[ii] = ii
                y_1[ii] = raw_P(p[ip],ii,n)
                simx[ii] = ii
                degree_count = 0
                for r in range(10000):
                    G = nx.erdos_renyi_graph(n, p[ip])
                    gmax = len(max(nx.connected_components(G), key=len))
                    if gmax == ii:
                        degree_count+=1

                sim_prob = degree_count / (10000)
                simy[ii] = sim_prob

            ax1.plot(x_1,y_1,label = p[ip])
            ax1.set_xlabel('i')
            ax1.set_ylabel("raw P")
            ax2.plot(simx,simy, label = p[ip])
            ax2.set_xlabel('i')
            ax2.set_ylabel("simulated P")
    #fig.legend()
    fig.savefig("bumpfig.png")


def error_graph(n,p):
    fig = plt.figure(figsize=(8, 8))
    x_array = np.zeros(n)
    y_array = np.zeros(n)
    for i in range(n):
        x_array[i] = i
        finiteRelS = finiteTheory.relSCurve(p, i, attack=True, fdict=fvals, pdict=pvals)
        print("extrema")
        print(scipy.signal.argrelextrema(finiteRelS, np.greater))
        #y_array[i] = scipy.signal.argrelextrema(finiteRelS, np.greater)[0][0]
        if len(scipy.signal.argrelextrema(finiteRelS, np.greater)[0]) == 0:
            y_array[i] == 0
        else:
            y_array[i] = scipy.signal.argrelextrema(finiteRelS, np.greater)[0][0]

    plt.plot(x_array, y_array)
    #plt.set_xlabel('n')
    #plt.set_ylabel("problematic % removed nodes")
    fig.savefig("errorfig.png")


#error_graph(20,.1)

#bump(20,20,[.7], "P")
#bump(20,20,.05, "kmax")



# def kmaxbump(n,p):
#     fig = plt.figure((8,8))
#     x_1 = np.zeros(n)
#     y_1 = np.zeros(n)
#     simx = np.zeros(i)
#     simy = np.zeros(i)
#     G = nx.erdos_renyi_graph(n, p[ip])
#     for i in range(n):
#         x_1[i] = i
#         y_1[i] = expectedMaxDegree(i, p)
