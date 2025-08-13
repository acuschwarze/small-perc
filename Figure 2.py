###############################################################################
#
# Code to create Figure 3/4 of paper
#   To create Figure 3, remove_bool = False
#   To create Figure 4, remove_bool = True
#
###############################################################################


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

# open precalculated values for recursion as part of finite theory S_rec
fvals = pickle.load(open('data/fvalues.p', 'rb'))
pvals = pickle.load(open('data/Pvalues.p', 'rb'))

# figure attributes
fig, axs = plt.subplots(1,2, figsize = [10,3.5])
colors = ['red','blue','orange','green','purple','cyan','magenta']
markers = ['o', 'x', 'v', 's', '+', 'd', '1']


############## panel b) ###################

# parameters
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
 

    # === Infinite Theory ===
    inf = infiniteTheory.relSCurve(n, p, attack=remove_bool,
                                   reverse=False, smooth_end=False)
    axs[0].plot(nodes_array, inf, color=colors[j])

    # === Finite Theory ===
    axs[0].plot(nodes_array, fin, linestyle='--', color=colors[j])
    axs[0].set(xlabel= "fraction " + r'$f$', ylabel='rel. LCC size')
    axs[0].set_ylim(-.1,1.05)
    pos2 = axs[0].get_position()
    axs[0].set_position([pos2.x0, pos2.y0, pos2.width, pos2.height])
    axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
          ncol=4) # fancybox=True, shadow=True



############## panel a) ###################

# parameters
n_threshold = .2 # percolation threshold for panel b)
nodes_list = [10,15,25,50] # nodes values for panel b)
probs_list = [(1/(n_threshold*(x-1))) for x in nodes_list] # calculated p values
remove_bool = True # False for Fig 3, True for Fig 4
simtrials = 100 # number of simulations

# node removal type
if remove_bool == True:
    remove_strat = "attack"
else: 
    remove_strat = 'random'


big_data = np.zeros((simtrials),dtype=object)

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
axs[1].legend()


########## final figure adjustments ##############
plt.subplots_adjust(left=0.06, right=.99, bottom=.15, top=0.90, wspace=.04)


axs[0].text(0.07, .1, '(a)', transform=axs[0].transAxes, fontsize=10, fontweight='normal', va='top', ha='right')
axs[1].text(0.07, .1, '(b)', transform=axs[1].transAxes, fontsize=10, fontweight='normal', va='top', ha='right')


if remove_bool == False:
    plt.savefig("Fig_3")
    plt.savefig("Fig_3.pdf")
else:
    plt.savefig("Fig_4")
    plt.savefig("Fig_4.pdf")