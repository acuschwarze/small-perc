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


# fig = plot_graphs(numbers_of_nodes=[20], edge_probabilities=[.1],
#                      graph_types=['ER'], remove_strategies=["attack"],
#                      performance='relative LCC', num_trials=100,
#                      smooth_end=False, forbidden_values=[], fdict=fvals, lcc_method_main="pmult", savefig='',
#                      simbool=True)

colors = ['red','blue','orange','green','purple','cyan','magenta']
n_threshold = .124 #.526
nodes_list = [10,15,20]
probs_list = [(1/(n_threshold*(x-1))) for x in nodes_list]
print("probs_list",probs_list)

remove_bool = True

for i_n in range(len(nodes_list)):
    n = nodes_list[i_n]
    nodes_array = np.arange(n)/n
    p = round(probs_list[i_n], 2)
    p_index = int(p/.01 - 1)
    all_sim = relSCurve_precalculated(n, p, targeted_removal=remove_bool, simulated=True, finite=False)
    #print("allsim",i,j,all_sim)
    sim = np.zeros(n)
    #print("i",i)
    for k in range(n):
        sim = sim + np.transpose(all_sim[:,k][:n])
    sim = sim / n
    
    fin = (relSCurve_precalculated(n, p, targeted_removal=remove_bool, simulated=False, finite=True)[:n])
    
    if i_n == len(nodes_list)-1:
        # path_name = "{}_attack{}_n{}.npy".format("infRelSCurve", remove_bool, n)
        # path_name = os.path.join("data", "synthetic_data", path_name)
        # all_inf = np.load(path_name)
        # #print(np.shape(all_inf))
        # inf = all_inf[p_index]
        inf = infiniteTheory.relSCurve(n, p, attack=remove_bool, smooth_end=False)
        plt.plot(nodes_array, inf, label = "infinite theory", color = "black")
    plt.plot(nodes_array, fin, label = "fin" + str(n), linestyle = '--', color = colors[i_n])
    plt.plot(nodes_array, sim, label = "sim" + str(n), marker = 'o', ms = 3, color = colors[i_n])
    #plt.set_title('Nwks with same percolation percentage: attack')
    # plt.set_xlabel(r'f')
    # plt.set_yticklabels([])
    # pos = plt.get_position()
    # plt.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
    # plt.legend(loc='center right', bbox_to_anchor=(2.25, 0.5))

#plt.legend(loc='center left',bbox_to_anchor=(1, 0.5))



plt.subplots_adjust(left=0.08, right=.8, bottom=.15, top=0.9, wspace=.1)
plt.savefig("Fig 2 test")

