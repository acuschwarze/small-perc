###############################################################################
#
# Code to create Figure 2 of paper
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

# parameters
n = 20
nodes = np.arange(n)/n
p = .1
p_index = int(p/.01 - 1)

# figure dimensions
fig, axs = plt.subplots(1,2, figsize = [6.5,3])

# booleans for node removal type: False = random removals, True = targeted removals
remove_bools = [False,True]

# number of simulations
simtrials=100

for i in range(len(remove_bools)):

    remove_bool = remove_bools[i]
    if remove_bool == True:
        remove = "Target"
        remove2 = "attack"
    elif remove_bool == False:
        remove = "Random"
        remove2 = "random"

    # get data for curve plotting
    big_data = np.zeros((simtrials),dtype=object)
    for j in range(len(nodes)):
        for k in range(simtrials):
                # simulated data
                sim_data = completeRCData(numbers_of_nodes=[n],
                                        edge_probabilities=[p], num_trials=1,
                                        performance='relative LCC', graph_types=['ER'],
                                        remove_strategies=[remove2])
                data_array = np.array(sim_data[0][0][0][0])

                # exclude the first row, because it is the number of nodes
                data_array = data_array[1:]

                # this can prevent bug about invalid values
                for val in []:
                    data_array[data_array == val] = np.nan

                # plot simulated data
                data = np.nanmean(data_array, axis=0)
                big_data[k] = data


    numtrials = len(data)

    # infinite theory S_inf values
    inf = infiniteTheory.relSCurve(n, p, attack=remove_bool, smooth_end=False)

    # finite theory S_rec from precalculated data
    fin_path = "{}_attack{}_n{}.npy".format("RelSCurve", remove_bool, n)
    fin_path = os.path.join("data", "synthetic_data", fin_path)
    all_fin = np.load(fin_path)
    fin = all_fin[p_index]


    # error bars for simulated data
    std_table = np.zeros(n)
    sim_y = np.zeros(n)

    for j in range(n):
        sim_data = np.zeros(numtrials)
        for i_nums in range(numtrials):
            sim_data[i_nums] = big_data[i_nums][j]
        std = np.std(sim_data)
        std_table[j] = std / 10 * 3 # 3 standard errors
        sim_y[j] = np.nanmean(sim_data)

    # plot labelling
    axs[i].errorbar(x=nodes, y=sim_y, yerr = std_table, marker = 'o', markersize=2.5, label = r"$\widebar{S}$", lw=1, color = "red")
    axs[i].plot(nodes, inf, label = r"${S}_{\infty}$", color = "black")
    axs[i].plot(nodes, fin, label = r"${S}_{rec}$", color = "blue", linestyle = '--')
    axs[i].set(xlabel= r'fraction $f$')
    if i==0:
        axs[i].set(ylabel= r'rel. LCC size')
    else:
        axs[i].set_yticklabels([])

# adjust legend and panel labels
axs[i].legend()
pos2 = axs[i].get_position()
axs[i].set_position([pos2.x0, pos2.y0, pos2.width, pos2.height])
axs[i].legend(loc='upper left', bbox_to_anchor=(.05, 1))
handles, labels = plt.gca().get_legend_handles_labels()
order = [2,0,1]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
axs[0].text(0.05, .9, '(a)', transform=axs[0].transAxes, fontsize=10, fontweight='normal', va='bottom', ha='left')
axs[1].text(0.05, .9, '(b)', transform=axs[1].transAxes, fontsize=10, fontweight='normal', va='bottom', ha='left')

plt.subplots_adjust(left=0.08, right=0.98, bottom=.15, top=0.99, wspace=.1)
plt.savefig("Figure_2.pdf")
