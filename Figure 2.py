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

fig, axs = plt.subplots(1,1, figsize = [6.4,3.5])

colors = ['red','blue','orange','green','purple','cyan','magenta']
n_threshold = .2
nodes_list = [10,15,20,25,50,75,100]
probs_list = [(1/(n_threshold*(x-1))) for x in nodes_list]
print(probs_list)
remove_bool = False

for i_n in range(len(nodes_list)):
    n = nodes_list[i_n]
    nodes_array = np.arange(n)/n
    p = round(probs_list[i_n], 2)
    p_index = int(p/.01 - 1)
    
    all_sim = relSCurve_precalculated(n, p, targeted_removal=False, simulated=True, finite=False)
    #print("allsim",i,j,all_sim)
    sim = np.zeros(n)
    #print("i",i)
    for k in range(n):
        sim = sim + np.transpose(all_sim[:,k][:n])
    sim = sim / n
    
    fin = (relSCurve_precalculated(n, p, targeted_removal=False, simulated=False, finite=True)[:n])
    
    axs.plot(nodes_array, fin, label = "finite theory: G(" + str(n) + "," + str(p) + ")", linestyle = '--', color = colors[i_n])
    axs.plot(nodes_array, sim, label = "simulations: G(" + str(n) + "," + str(p) + ")", marker = "o", ms = 3, color = colors[i_n])
    #axs[0].set_title('Nwks with same percolation percentage: random')
    axs.set(xlabel=r'$f$', ylabel=r'$\langle S \rangle$')
    #axs.margins(x=[.08,3])

    if i_n == len(nodes_list)-1:
            path_name = "{}_attack{}_n{}.npy".format("infRelSCurve", remove_bool, n)
            path_name = os.path.join("data", "synthetic_data", path_name)
            all_inf = np.load(path_name)
            #print(np.shape(all_inf))
            inf = all_inf[p_index]
            axs.plot(nodes_array, inf, label = "infinite theory")

    pos = axs.get_position()
    axs.set_position([pos.x0, pos.y0, pos.width, pos.height])
    axs.legend(loc='upper right', bbox_to_anchor=(1.48, 1.15),prop={'size': 7.8})


#plt.legend(loc='center left',bbox_to_anchor=(1, 0.5))
plt.subplots_adjust(left=0.1, right=.7, bottom=.15, top=0.9, wspace=3)
#rect = .08,.15,8,.9
#ax = fig.add_axes(rect)
plt.savefig("Fig 2 final")


# colors = ['red','blue','orange','green','purple','cyan','magenta']
# n_threshold = .2
# nodes_list = [10,15,20,25,50,75,100]
# probs_list = [(1/(n_threshold*(x-1))) for x in nodes_list]
# print(probs_list)
# remove_bool = False

# for i_n in range(len(nodes_list)):
#     n = nodes_list[i_n]
#     nodes_array = np.arange(n)/n
#     p = round(probs_list[i_n], 2)
#     p_index = int(p/.01 - 1)
    
#     all_sim = relSCurve_precalculated(n, p, targeted_removal=False, simulated=True, finite=False)
#     #print("allsim",i,j,all_sim)
#     sim = np.zeros(n)
#     #print("i",i)
#     for k in range(n):
#         sim = sim + np.transpose(all_sim[:,k][:n])
#     sim = sim / n
    
#     fin = (relSCurve_precalculated(n, p, targeted_removal=False, simulated=False, finite=True)[:n])
    
#     axs[0].plot(nodes_array, fin, label = "fin" + str(n), linestyle = '--', color = colors[i_n])
#     axs[0].plot(nodes_array, sim, label = "sim"+str(n), marker = "o", ms = 3, color = colors[i_n])
#     #axs[0].set_title('Nwks with same percolation percentage: random')
#     axs[0].set(xlabel=r'$f$', ylabel=r'$\langle S \rangle$')

#     if i_n == len(nodes_list)-1:
#             path_name = "{}_attack{}_n{}.npy".format("infRelSCurve", remove_bool, n)
#             path_name = os.path.join("data", "synthetic_data", path_name)
#             all_inf = np.load(path_name)
#             #print(np.shape(all_inf))
#             inf = all_inf[p_index]
#             axs[0].plot(nodes_array, inf, label = "infinite theory")

#     pos = axs[0].get_position()
#     axs[0].set_position([pos.x0, pos.y0, pos.width, pos.height])
#     axs[0].legend(loc='upper right', bbox_to_anchor=(2.65, 1.15))

# nodes_list = [10,15,20,25,50]
# probs_list = [(1/(n_threshold*(x-1))) for x in nodes_list]
# print(probs_list)
# remove_bool = True

# for i_n in range(len(nodes_list)):
#     n = nodes_list[i_n]
#     nodes_array = np.arange(n)/n
#     p = round(probs_list[i_n], 2)
#     p_index = int(p/.01 - 1)
#     all_sim = relSCurve_precalculated(n, p, targeted_removal=remove_bool, simulated=True, finite=False)
#     #print("allsim",i,j,all_sim)
#     sim = np.zeros(n)
#     #print("i",i)
#     for k in range(n):
#         sim = sim + np.transpose(all_sim[:,k][:n])
#     sim = sim / n
    
#     fin = (relSCurve_precalculated(n, p, targeted_removal=remove_bool, simulated=False, finite=True)[:n])
    
#     axs[1].plot(nodes_array, fin, label = "fin" + str(n), linestyle = '--', color = colors[i_n])
#     axs[1].plot(nodes_array, sim, label = "sim" + str(n), marker = 'o', ms = 3, color = colors[i_n])
#     #axs[1].set_title('Nwks with same percolation percentage: attack')
#     axs[1].set(xlabel=r'$f$')
#     axs[1].set_yticklabels([])
#     # pos = axs[1].get_position()
#     # axs[1].set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
#     # axs[1].legend(loc='center right', bbox_to_anchor=(2.25, 0.5))

#     if i_n == len(nodes_list)-1:
#         # path_name = "{}_attack{}_n{}.npy".format("infRelSCurve", remove_bool, n)
#         # path_name = os.path.join("data", "synthetic_data", path_name)
#         # all_inf = np.load(path_name)
#         # #print(np.shape(all_inf))
#         # inf = all_inf[p_index]
#         inf = infiniteTheory.relSCurve(n, p, attack=remove_bool, smooth_end=False)
#         axs[1].plot(nodes_array, inf, label = "infinite theory")

# #plt.legend(loc='center left',bbox_to_anchor=(1, 0.5))
# plt.subplots_adjust(left=0.08, right=.8, bottom=.15, top=0.9, wspace=.1)
# plt.savefig("Fig 2 final")





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
