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
colors = ['red','blue','orange','green','purple','cyan','magenta']
markers = ['o', 'x', 'v', 's', '+', 'd', '1']
n_threshold = .2
nodes_list = [10,15,25,50]
probs_list = [(1/(n_threshold*(x-1))) for x in nodes_list]
print(probs_list)
remove_bool = True
simtrials = 100

if remove_bool == True:
    remove_strat = "attack"
else: 
    remove_strat = 'random'

big_data = np.zeros((simtrials),dtype=object)

# for i_n in range(len(nodes_list)):
#     n = nodes_list[i_n]
#     nodes_array = np.arange(n)/n
#     p = round(probs_list[i_n], 2)
#     p_index = int(p/.01 - 1)
    
    
#     if n == 10 or n==15 or n > 100:
#         sim = np.zeros(n)
#         #for j in range(simtrials):
#         sim = completeRCData(numbers_of_nodes=[n],
#                                   edge_probabilities=[p], num_trials=1000,
#                                   performance='relative LCC', graph_types=['ER'],
#                                   remove_strategies=[remove_strat])
#         data_array = np.array(sim[0][0][0][0])

#         # exclude the first row, because it is the number of nodes
#         data_array = data_array[1:]

#         # this can prevent some sort of bug about invalid values
#         for val in []:
#             data_array[data_array == val] = np.nan

#         # plot simulated data
#         #removed_fraction = np.arange(n) / n
#         line_data = np.nanmean(data_array, axis=0)
#         sim = line_data
        
        
#         #sim /= simtrials
#     #elif n > 100:
#         #sim = finiteTheory.relSCurve(p, n,
#                 #                        attack=remove_bool, fdict=fvals, pdict=pvals, lcc_method_relS="pmult", executable_path='libs/p-recursion-float128.exe')
        
#     else:
#         all_sim = relSCurve_precalculated(n, p, targeted_removal=remove_bool, simulated=True, finite=False)
#     #print("allsim",i,j,all_sim)
#         sim = np.zeros(n)
#         #print("i",i)
#         for k in range(n):
#             sim = sim + np.transpose(all_sim[:,k][:n])
#         sim = sim / n
    
#     if n > 100:
#         fin = finiteTheory.relSCurve(p, n,
#             attack=remove_bool, fdict=fvals, pdict=pvals, lcc_method_relS="pmult", executable_path='libs/p-recursion-float128.exe')

#     else:
#         fin = (relSCurve_precalculated(n, p, targeted_removal=remove_bool, simulated=False, finite=True)[:n])
    
#     axs[1].plot(nodes_array, fin, linestyle = '--', color = colors[i_n])
#     axs[1].plot(nodes_array, sim, label = r"$N = $"+str(n), linestyle = ' ', marker = markers[i_n], ms = 3, color = colors[i_n])
#     #axs[0].set_title('Nwks with same percolation percentage: random')
#     axs[1].set_yticklabels([])
#     axs[1].set_ylim(-.1,1.05)
#     axs[1].set(xlabel=r'fraction $f$')
#     #axs[1].text(-0.1, 1.15, "B", transform=axs[1].transAxes,fontweight="bold",fontsize=16, va='top', ha='right')


#     if i_n == len(nodes_list)-1:
#             path_name = "{}_attack{}_n{}.npy".format("infRelSCurve", remove_bool, n)
#             path_name = os.path.join("data", "synthetic_data", path_name)
#             all_inf = np.load(path_name)
#             #print(np.shape(all_inf))
#             #inf = all_inf[p_index]
#             inf = infiniteTheory.relSCurve(n, p, attack=remove_bool, reverse=False, smooth_end=False)
#             axs[1].plot(nodes_array, inf, label = r"${\langle S \rangle}_{N \to \infty}$", color = "black")
#     pos = axs[1].get_position()
#     axs[1].set_position([pos.x0, pos.y0, pos.width, pos.height])
#     #axs[1].legend(loc="upper left",ncol=6)
#     axs[1].legend()
#     # axs[1].legend(loc='upper right', bbox_to_anchor=(1.35, 1.15))


#     for j in range(len(nodes_array)):
#             for k in range(simtrials):
#                     sim = completeRCData(numbers_of_nodes=[n],
#                                     edge_probabilities=[p], num_trials=1000,
#                                     performance='relative LCC', graph_types=['ER'],
#                                     remove_strategies=[remove_strat])
#                     data_array = np.array(sim[0][0][0][0])

#                     # exclude the first row, because it is the number of nodes
#                     data_array = data_array[1:]
#                     #print("data_array", data_array)

#                     # this can prevent some sort of bug about invalid values
#                     for val in []:
#                         data_array[data_array == val] = np.nan

#                     # plot simulated data
#                     data = np.nanmean(data_array, axis=0)
#                     big_data[k] = data
#                     #print("final big_data", big_data)

        
#     std_table = np.zeros(n)
#     sim_y = np.zeros(n)

#     print("bigtdata dimesnions", big_data.shape)
#     numtrials = n
#     for j in range(n):
#         sim_data = np.zeros(numtrials)
#         for i_nums in range(numtrials):
#             sim_data[i_nums] = big_data[i_nums][j]
#             #if i_nums == 0:
#                 #print(big_data[i_nums][j])
#         print("sim_data dimesnions", sim_data.shape)
#         std = np.std(sim_data)
#         print(" std", std)
#         #print(std)
#         std_table[j] = std / 10 * 1000 # 10 for standard error (sqrt100)
#         sim_y[j] = np.nanmean(sim_data)
#     print("std_tablew")
#     print(std_table)

#     axs[1].errorbar(x=nodes_array, y=sim_y, yerr = std_table, marker = 'o', markersize=2.5, lw=1, color = colors[i_n])

for i_n, n in enumerate(nodes_list):
    nodes_array = np.arange(n) / n
    p = round(probs_list[i_n], 2)
    p_index = int(p / .01 - 1)
    big_data = np.zeros((simtrials, n))  # store LCC over f for each trial

    # Simulate data once for all trials
    for k in range(simtrials):
        sim = completeRCData(numbers_of_nodes=[n],
                             edge_probabilities=[p], num_trials=1,
                             performance='relative LCC', graph_types=['ER'],
                             remove_strategies=[remove_strat])
        data_array = np.array(sim[0][0][0][0])[1:]  # exclude first row
        big_data[k] = np.nan_to_num(data_array)

    # Compute mean and std
    sim_mean = np.nanmean(big_data, axis=0)
    sim_std = np.nanstd(big_data, axis=0) / np.sqrt(simtrials) * 3  # 3-sigma bounds

    # Plot simulated data with error bars
    axs[1].errorbar(x=nodes_array, y=sim_mean, yerr=sim_std,
                    marker=markers[i_n], markersize=3, lw=1,
                    color=colors[i_n], label="_nolegend_")

    axs[1].plot(nodes_array, sim_mean,
            linestyle=' ', marker=markers[i_n], color=colors[i_n],
            label=f"$N$={n}", markersize=3)

    # Plot finite theory
    if n > 100:
        fin = finiteTheory.relSCurve(p, n, attack=remove_bool, fdict=fvals,
                                     pdict=pvals, lcc_method_relS="pmult",
                                     executable_path='libs/p-recursion-float128.exe')
    else:
        fin = relSCurve_precalculated(n, p, targeted_removal=remove_bool,
                                      simulated=False, finite=True)[:n]

    axs[1].plot(nodes_array, fin, linestyle='--', color=colors[i_n])

    # Plot infinite theory (only once)
    if i_n == len(nodes_list) - 1:
        inf = infiniteTheory.relSCurve(n, p, attack=remove_bool,
                                       reverse=False, smooth_end=False)
        axs[1].plot(nodes_array, inf, color="black", label=r"$S_{\infty}$")

axs[1].set_yticklabels([])
axs[1].set_ylim(-.1,1.05)
axs[1].set(xlabel=r'fraction $f$')

pos = axs[1].get_position()
axs[1].set_position([pos.x0, pos.y0, pos.width, pos.height])
#axs[1].legend(loc="upper left",ncol=6)
axs[1].legend()




# n = 25
# mult_probs = [.05,.1,.3, 1]
# nodes_array = np.arange(n)/n
# trials = 1
# for j in range(len(mult_probs)):
#     p_index = int(mult_probs[j]/.01 - 1)
#     # axs[1].plot(np.arange(n)/n, finiteTheory.relSCurve(mult_probs[j], n,
#     #                                         attack=False, fdict=fvals, pdict=pvals, lcc_method_relS="pmult", executable_path='libs/p-recursion-float128.exe'),
#     #                                         label = r"$p=$"+str(mult_probs[j]))

    
#     fin = (relSCurve_precalculated(n, mult_probs[j], targeted_removal=remove_bool, simulated=False, finite=True)[:n])

#     if mult_probs[j] == .05 or mult_probs[j] ==.1:
#         sim = completeRCData(numbers_of_nodes=[n],
#                                   edge_probabilities=[mult_probs[j]], num_trials=1000,
#                                   performance='relative LCC', graph_types=['ER'],
#                                   remove_strategies=[remove_strat])
#         data_array = np.array(sim[0][0][0][0])

#         # exclude the first row, because it is the number of nodes
#         data_array = data_array[1:]

#         # this can prevent some sort of bug about invalid values
#         for val in []:
#             data_array[data_array == val] = np.nan

#         # plot simulated data
#         #removed_fraction = np.arange(n) / n
#         line_data = np.nanmean(data_array, axis=0)
#         sim = line_data
#     else:
#         all_sim = relSCurve_precalculated(n, mult_probs[j], targeted_removal=remove_bool, simulated=True, finite=False)
#         #print("allsim",i,j,all_sim)
#         sim = np.zeros(n)
#         #print("i",i)
#         for k in range(n):
#             sim = sim + np.transpose(all_sim[:,k][:n])
#         sim = sim / n

#     path_name = "{}_attack{}_n{}.npy".format("infRelSCurve", remove_bool, n)
#     path_name = os.path.join("data", "synthetic_data", path_name)
#     all_inf = np.load(path_name)
#     #print(np.shape(all_inf))
#     #inf = all_inf[p_index]
#     inf = infiniteTheory.relSCurve(n, mult_probs[j], attack=remove_bool, reverse=False, smooth_end=False)
#     axs[0].plot(nodes_array, inf, color = colors[j])

#     axs[0].plot(nodes_array, fin, linestyle = '--', color = colors[j])
#     axs[0].plot(nodes_array, sim, label = r"$p = $" + str(mult_probs[j]), linestyle = " ", marker = markers[j], ms = 3, color = colors[j])
    
    
n = 25
mult_probs = [.05, .1, .3, 1] #[.05, .1, .3, 1]
nodes_array = np.arange(n) / n
simtrials = 100  # smaller for demo; increase for more reliable error estimates

for j, p in enumerate(mult_probs):
    p_index = int(p / .01 - 1)

    # === Get Finite Theory Curve ===
    fin = relSCurve_precalculated(n, p, targeted_removal=remove_bool,
                                  simulated=False, finite=True)[:n]

    # === Get Simulation Curve (with error bars) ===
    big_data = np.zeros((simtrials, n))

    if p <= 0.1:
        for trial in range(simtrials):
            sim = completeRCData(numbers_of_nodes=[n],
                                 edge_probabilities=[p], num_trials=1,
                                 performance='relative LCC', graph_types=['ER'],
                                 remove_strategies=[remove_strat])
            data_array = np.array(sim[0][0][0][0])[1:]  # skip first row (n)
            big_data[trial] = np.nan_to_num(data_array)
            print("data", data_array.shape)
            print(data_array)

        sim_mean = np.mean(big_data, axis=0)
        sim_stderr = np.std(big_data, axis=0) / np.sqrt(simtrials) * 3

        # Plot simulation with error bars
        axs[0].errorbar(nodes_array, sim_mean, yerr=sim_stderr,
                        marker=markers[j], markersize=3, linestyle=' ',
                        linewidth=1, color=colors[j], label="_nolegend_") #label=fr"$p = {p}$")
        
        axs[0].plot(nodes_array, sim_mean,
            linestyle=' ', marker=markers[j], color=colors[j],
            label=fr"$p = {p}$", markersize=3)


    else:
        all_sim = relSCurve_precalculated(n, p, targeted_removal=remove_bool,
                                          simulated=True, finite=False)
        sim = np.zeros(n)

        sim_mean = np.mean(np.transpose(all_sim), axis=0)
        sim_stderr = np.std(np.transpose(all_sim), axis=0) / np.sqrt(simtrials) * 3

        # Plot simulation with error bars
        axs[0].errorbar(nodes_array, sim_mean, yerr=sim_stderr,
                        marker=markers[j], markersize=3, linestyle=' ',
                        linewidth=1, color=colors[j], label="_nolegend_")  
        axs[0].plot(nodes_array, sim_mean,
            linestyle=' ', marker=markers[j], color=colors[j],
            label=fr"$p = {p}$", markersize=3)
 

        # axs[0].plot(nodes_array, sim, linestyle=' ', marker=markers[j],
        #             ms=3, color=colors[j], label=fr"$p = {p}$")

    # === Infinite Theory ===
    inf = infiniteTheory.relSCurve(n, p, attack=remove_bool,
                                   reverse=False, smooth_end=False)
    axs[0].plot(nodes_array, inf, color=colors[j])

    # === Finite Theory ===
    axs[0].plot(nodes_array, fin, linestyle='--', color=colors[j])

    
    
    #axs[1].set_title('Nwks with same percolation percentage: attack')
    axs[0].set(xlabel= "fraction " + r'$f$', ylabel='rel. LCC size')
    axs[0].set_ylim(-.1,1.05)
    pos2 = axs[0].get_position()
    axs[0].set_position([pos2.x0, pos2.y0, pos2.width, pos2.height])
    axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
          ncol=4) # fancybox=True, shadow=True
    #axs[0].legend(loc="upper left",ncol=4)
    
    #axs[0].text(-0.1, 1.15, "A", transform=axs[1].transAxes,fontweight="bold",fontsize=16, va='top', ha='right')
    # axs[0].legend(loc='center right', bbox_to_anchor=(-.05, 0.11))

    # pos = axs[1].get_position()
    # axs[1].set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
    # axs[1].legend(loc='center right', bbox_to_anchor=(2.25, 0.5))

    # if i_n == len(nodes_list)-1:
    #     # path_name = "{}_attack{}_n{}.npy".format("infRelSCurve", remove_bool, n)
    #     # path_name = os.path.join("data", "synthetic_data", path_name)
    #     # all_inf = np.load(path_name)
    #     # #print(np.shape(all_inf))
    #     # inf = all_inf[p_index]
    #     inf = infiniteTheory.relSCurve(n, p, attack=remove_bool, smooth_end=False)
    #     axs[1].plot(nodes_array, inf, label = "infinite theory")



#plt.legend(loc='center left',bbox_to_anchor=(1, 0.5))
plt.subplots_adjust(left=0.06, right=.99, bottom=.15, top=0.90, wspace=.04)


axs[0].text(0.07, .1, '(a)', transform=axs[0].transAxes, fontsize=10, fontweight='normal', va='top', ha='right')
axs[1].text(0.07, .1, '(b)', transform=axs[1].transAxes, fontsize=10, fontweight='normal', va='top', ha='right')


if remove_bool == False:
    plt.savefig("Fig 2 final")
    plt.savefig("Fig_2_final.pdf")
else:
    plt.savefig("Fig 2 final attack")
    plt.savefig("Fig_2_final_attack.pdf")