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

fig, axs = plt.subplots(1,2, figsize = [10,3.5])

colors = ['red','blue','orange','green','purple','cyan']
n_threshold = .2
nodes_list = [10,15,25,50,75,100]
probs_list = [(1/(n_threshold*(x-1))) for x in nodes_list]
print(probs_list)
remove_bool = False
num_trials = 100

for i_n in range(len(nodes_list)):
    n = nodes_list[i_n]
    nodes_array = np.arange(n)/n
    p = round(probs_list[i_n], 2)
    p_index = int(p/.01 - 1)
    
    all_sim = relSCurve_precalculated(n, p, targeted_removal=False, simulated=True, finite=False, num_trials=num_trials)
    print("all_sim.shape", all_sim.shape)
    #print(all_sim[:,:4])
    #print("allsim",i,j,all_sim)
    sim = np.zeros(n)
    #print("i",i)
    for k in range(num_trials):
        sim = sim + np.transpose(all_sim[:,k][:n])
    sim = sim / num_trials
    
    fin = (relSCurve_precalculated(n, p, targeted_removal=False, simulated=False, finite=True))
    #print('fin.shape', fin.shape)
    fin = fin[:n]
    #print('fin.shape2', fin.shape)
    
    if i_n == len(nodes_list)-1:
        path_name = "{}_attack{}_n{}.npy".format("infRelSCurve", remove_bool, n)
        path_name = os.path.join("data", "synthetic_data", path_name)
        all_inf = np.load(path_name)
        #print(np.shape(all_inf))
        inf = all_inf[p_index]
        axs[0].plot(nodes_array, inf, label = "infinite theory")
    axs[0].plot(nodes_array, fin, label = "fin" + str(n), linestyle = '--', color = colors[i_n])
    axs[0].plot(nodes_array, sim, label = "sim"+str(n), marker = "o", ms = 3, color = colors[i_n])
    #axs[0].set_title('Nwks with same percolation percentage: random')
    axs[0].set(xlabel='nodes', ylabel='Relative LCC')
    pos = axs[0].get_position()
    axs[0].set_position([pos.x0, pos.y0, pos.width, pos.height])
    axs[0].legend(loc='upper right', bbox_to_anchor=(2.65, 1.15))

nodes_list = [10,15,25,50]
probs_list = [(1/(n_threshold*(x-1))) for x in nodes_list]
print(probs_list)
remove_bool = True

for i_n in range(len(nodes_list)):
    n = nodes_list[i_n]
    nodes_array = np.arange(n)/n
    p = round(probs_list[i_n], 2)
    p_index = int(p/.01 - 1)
    all_sim = relSCurve_precalculated(n, p, targeted_removal=remove_bool, simulated=True, finite=False, num_trials=num_trials)
    #print("allsim",i,j,all_sim)
    #print(all_sim[:,:4])
    sim = np.zeros(n)
    #print("i",i)
    for k in range(num_trials):
        sim = sim + np.transpose(all_sim[:,k][:n])
    sim = sim / num_trials
    
    fin = (relSCurve_precalculated(n, p, targeted_removal=remove_bool, simulated=False, finite=True))
    #print('fin.shape', fin.shape)
    fin = fin[:n]
    #print('fin.shape2', fin.shape)

    if i_n == len(nodes_list)-1:
        # path_name = "{}_attack{}_n{}.npy".format("infRelSCurve", remove_bool, n)
        # path_name = os.path.join("data", "synthetic_data", path_name)
        # all_inf = np.load(path_name)
        # #print(np.shape(all_inf))
        # inf = all_inf[p_index]
        inf = infiniteTheory.relSCurve(n, p, attack=remove_bool, smooth_end=False)
        axs[1].plot(nodes_array, inf, label = "infinite theory")
    axs[1].plot(nodes_array, fin, label = "fin" + str(n), linestyle = '--', color = colors[i_n])
    axs[1].plot(nodes_array, sim, label = "sim" + str(n), marker = 'o', ms = 3, color = colors[i_n])
    #axs[1].set_title('Nwks with same percolation percentage: attack')
    axs[1].set(xlabel='nodes')
    axs[1].set_yticklabels([])
    # pos = axs[1].get_position()
    # axs[1].set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
    # axs[1].legend(loc='center right', bbox_to_anchor=(2.25, 0.5))

#plt.legend(loc='center left',bbox_to_anchor=(1, 0.5))
plt.subplots_adjust(left=0.08, right=.8, bottom=.15, top=0.9, wspace=.1)
plt.savefig("Fig 2 final")





# colors = ['red','blue','orange','green','purple','cyan']
# fig = plt.figure(figsize=(8, 8))
# perc_rand = pd.read_pickle("percolation_random")
# nodes = perc_rand.nodes.values
# data = perc_rand.loc[:,"fin theory RLCC"]
# sim = perc_rand.loc[:,'simulated RLCC']
# n = len(perc_rand.nodes.values)
# for j in range(n):
#     nodes_array = np.arange(nodes[j], dtype=float)/nodes[j]
#     plt.plot(nodes_array, data[j], color = colors[j])
#     plt.plot(nodes_array,sim[j],'o',color = colors[j])
# p = 1/(.2*(100-1))
# plt.plot(nodes_array, infiniteTheory.relSCurve(100, p,
#                             attack=False, smooth_end=False), label = "inf theory")
# plt.xlabel("percent nodes removed")
# plt.ylabel("relative LCC")
# plt.title("relative LCC over nodes removed randomly")
# plt.savefig("one_perc_graph_random")


# colors = ['red','blue','orange','green','purple','cyan']
# fig = plt.figure(figsize=(8, 8))
# perc_att = pd.read_pickle("percolation_attack")
# #print(perc_att)
# #print(perc_att.iloc[5][2])
# #print(perc_att.iloc[5][3])
# nodes = perc_att.nodes.values
# data = perc_att.loc[:,"fin theory RLCC"]
# sim = perc_att.loc[:,'simulated RLCC']
# n = len(perc_att.nodes.values)
# for j in range(n-2):
#     nodes_array = np.arange(nodes[j], dtype=float)/nodes[j]
#     plt.plot(nodes_array, data[j], color = colors[j])
#     plt.plot(nodes_array,sim[j],'o',color = colors[j])
# p = 1/(.2*(50-1))
# plt.plot(nodes_array, infiniteTheory.relSCurve(50, p,
#                             attack=True, smooth_end=False), label = "inf theory")
# plt.xlabel("percent nodes removed")
# plt.ylabel("relative LCC")
# plt.title("relative LCC over nodes removed targeted")
# plt.savefig("one_perc_graph_attack")
