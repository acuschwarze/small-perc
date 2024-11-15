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

# random removals
n=20
nodes = np.arange(n)/n
p=.1
p_index = int(p/.01 - 1)


fig, axs = plt.subplots(1,2, figsize = [6.5,3])

remove_bools = [False,True]

simtrials=100

for i in range(len(remove_bools)):

    remove_bool = remove_bools[i]
    if remove_bool == True:
        remove = "Target"
        remove2 = "attack"
    elif remove_bool == False:
        remove = "Random"
        remove2 = "random"
    #data = relSCurve_precalculated(n, p, targeted_removal=remove_bool, simulated=True, finite=False)
    big_data = np.zeros((simtrials),dtype=object)
    for j in range(len(nodes)):
        for k in range(simtrials):
                sim_data = completeRCData(numbers_of_nodes=[n],
                                        edge_probabilities=[p], num_trials=1,
                                        performance='relative LCC', graph_types=['ER'],
                                        remove_strategies=[remove2])
                data_array = np.array(sim_data[0][0][0][0])

                # exclude the first row, because it is the number of nodes
                data_array = data_array[1:]

                # this can prevent some sort of bug about invalid values
                for val in []:
                    data_array[data_array == val] = np.nan

                # plot simulated data
                data = np.nanmean(data_array, axis=0)
                big_data[k] = data


    #sim_path = "{}_attack{}_n{}.npy".format("simRelSCurve", remove_bool, n)
    #sim_path = os.path.join("data", "synthetic_data", sim_path)
    #data = np.load(sim_path)
    print(data)
    print(np.shape(data))
    numtrials = len(data)

    inf = infiniteTheory.relSCurve(n, p, attack=remove_bool, smooth_end=False)

    fin_path = "{}_attack{}_n{}.npy".format("RelSCurve", remove_bool, n)
    fin_path = os.path.join("data", "synthetic_data", fin_path)
    all_fin = np.load(fin_path)
    fin = all_fin[p_index]


    std_table = np.zeros(n)
    sim_y = np.zeros(n)

    # for j in range(n):
    #     sim_data = np.zeros(numtrials)
    #     for i_nums in range(numtrials):
    #         sim_data[i_nums] = data[j][i_nums]
    #     std = np.std(sim_data)
    #     #print(std)
    #     std_table[j] = std / 10 * 3 # 10 for standard error (sqrt100)
    #     sim_y[j] = np.nanmean(sim_data)

    for j in range(n):
        sim_data = np.zeros(numtrials)
        for i_nums in range(numtrials):
            sim_data[i_nums] = big_data[i_nums][j]
        std = np.std(sim_data)
        #print(std)
        std_table[j] = std / 10 * 3 # 10 for standard error (sqrt100)
        sim_y[j] = np.nanmean(sim_data)

    axs[i].errorbar(x=nodes, y=sim_y, yerr = std_table, marker = 'o', markersize=2.5, label = r"$\widebar{S}$", lw=1, color = "red")
    axs[i].plot(nodes, inf, label = r"${\langle S \rangle}_{N \to \infty}$", color = "black")
    axs[i].plot(nodes, fin, label = r"${\langle S \rangle}$", color = "blue", linestyle = '--')
    #axs[i].set_title("Fin/Inf Theory: n=" + str(n) + ", p=" + str(p) + ", removal " + str(remove))
    axs[i].set(xlabel= r'fraction $f$')
    if i==0:
        axs[i].set(ylabel= r'rel. LCC size')
    else:
        axs[i].set_yticklabels([])

axs[i].legend()
pos2 = axs[i].get_position()
axs[i].set_position([pos2.x0, pos2.y0, pos2.width, pos2.height])
axs[i].legend(loc='upper left', bbox_to_anchor=(.05, 1))
handles, labels = plt.gca().get_legend_handles_labels()
order = [2,0,1]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])

#axs[i].legend(loc='center right', bbox_to_anchor=(0, 0.1))

plt.subplots_adjust(left=0.08, right=0.98, bottom=.15, top=0.99, wspace=.1)
plt.savefig("Fig_1_final.pdf")

# inf_rand_path = "{}_attack{}_n{}.npy".format("infRelSCurve", remove_bool, n)
# inf_rand_path = os.path.join("data", "synthetic_data", inf_targ_path)
# all_inf_rand = np.load(inf_targ_path)
# #inf_05 = all_inf[p_index]
# inf = infiniteTheory.relSCurve(n, p, attack=True, smooth_end=False)

# fin_targ_path = "{}_attack{}_n{}.npy".format("infRelSCurve", remove_bool, n)
# fin_targ_path = os.path.join("data", "synthetic_data", fin_targ_path)
# all_fin = np.load(fin_targ_path)
# fin = all_fin[p_index]


# data = np.load(r"C:\Users\jj\Downloads\GitHub\small-perc\data\synthetic_data\p0.05\simRelSCurve_attackFalse_n100_p0.05.npy")

# numtrials = len(data)
# n = len(data[0])
# nodes = np.arange(n)/n

# p = .05
# p_index = int(p/.01 - 1)

# all_inf = np.load(r"C:\Users\jj\Downloads\GitHub\small-perc\data\synthetic_data\infRelSCurve_attackFalse_n100.npy")
# inf_05 = all_inf[p_index]

# all_fin = np.load(r"C:\Users\jj\Downloads\GitHub\small-perc\data\synthetic_data\RelSCurve_attackFalse_n100.npy")
# fin_05 = all_fin[p_index]

# std_table = np.zeros(n)
# sim_y = np.zeros(n)

# for j in range(n):
#     sim_data = np.zeros(numtrials)
#     for i in range(numtrials):
#         sim_data[i] = data[i][j]
#     std = np.std(sim_data)
#     std_table[j] = std
#     sim_y[j] = np.nanmean(sim_data)


# plt.errorbar(x=nodes, y=sim_y, yerr = std_table, label = "simulations")
# plt.plot(nodes, inf_05, label = "infinite theory")
# plt.plot(nodes, fin_05, label = "finite theory")
# plt.xlabel("percent nodes removed")
# plt.ylabel("Relative LCC")
# plt.title("Simulations of Randomal Removal vs Fin/Inf Theory: n=" + str(n) + " and p=" + str(p))
# plt.legend()
# #plt.savefig("Fig 1a random theory and error - n=" + str(n) + ", p=05")


# # attack removals
# remove_bool = True
# n=20
# p=.1
# data = relSCurve_precalculated(n, p, targeted_removal=False, simulated=True, finite=False)
# #print(data)
# # data = relSCurve_precalculated(n, p, targeted_removal=True, simulated=False, finite=True)
# # print(data)
# #print(np.shape(data))
# #data = np.load(r"C:\Users\jj\Downloads\GitHub\small-perc\data\synthetic_data\p0.05\simRelSCurve_attackTrue_n100_p0.05.npy")

# numtrials = len(data[0])
# #print(numtrials)
# #n = len(data[0])
# nodes = np.arange(n)/n

# p_index = int(p/.01 - 1)


# # inf_targ_path = "{}_attack{}_n{}.npy".format("infRelSCurve", remove_bool, n)
# # inf_targ_path = os.path.join("data", "synthetic_data", inf_targ_path)
# # all_inf_targ = np.load(inf_targ_path)
# #inf_05 = all_inf[p_index]
# inf_targ = infiniteTheory.relSCurve(n, p, attack=True, smooth_end=False)

# fin_targ_path = "{}_attack{}_n{}.npy".format("infRelSCurve", remove_bool, n)
# fin_targ_path = os.path.join("data", "synthetic_data", fin_targ_path)
# all_fin_targ = np.load(fin_targ_path)
# fin_targ = all_fin[p_index]

# std_table_targ = np.zeros(n)
# sim_y_targ = np.zeros(n)

# for j in range(n):
#     sim_data = np.zeros(numtrials)
#     for i in range(numtrials):
#         sim_data[i] = data[j][i]
#     std = np.std(sim_data)
#     #print(std)
#     std_table[j] = std
#     sim_y[j] = np.nanmean(sim_data)

# #print(std_table)
# plt.errorbar(x=nodes, y=sim_y, yerr = std_table, label = "simulations")
# plt.plot(nodes, inf, label = "infinite theory")
# plt.plot(nodes, fin, label = "finite theory")
# plt.xlabel("percent nodes removed")
# plt.ylabel("Relative LCC")
# plt.title("Simulations of Targeted Removal vs Fin/Inf Theory: n=" + str(n) + " and p=" + str(p))
# plt.legend()
# plt.savefig("Fig 1b targeted theory and error - n=" + str(n) + ", p=05")

