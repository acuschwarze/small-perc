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
markers = ['o', 'x', 'v', '+', 's', 'd', '1']
n_threshold = .2
nodes_list = [10,15,50]
probs_list = [(1/(n_threshold*(x-1))) for x in nodes_list]
print(probs_list)
remove_bool = True

for i_n in range(len(nodes_list)):
    n = nodes_list[i_n]
    nodes_array = np.arange(n)/n
    p = round(probs_list[i_n], 2)
    p_index = int(p/.01 - 1)
    
    
    if n == 10 or n==15:
        sim = np.zeros(n)
        for j in range(100):
            sim += finiteTheory.relSCurve(p, n,
                                        attack=remove_bool, fdict=fvals, pdict=pvals, lcc_method_relS="pmult", executable_path='libs/p-recursion-float128.exe')
        sim /= 100
    else:
        all_sim = relSCurve_precalculated(n, p, targeted_removal=remove_bool, simulated=True, finite=False)
    #print("allsim",i,j,all_sim)
        sim = np.zeros(n)
        #print("i",i)
        for k in range(n):
            sim = sim + np.transpose(all_sim[:,k][:n])
        sim = sim / n
    
    fin = (relSCurve_precalculated(n, p, targeted_removal=remove_bool, simulated=False, finite=True)[:n])
    
    axs[1].plot(nodes_array, fin, linestyle = '--', color = colors[i_n])
    axs[1].plot(nodes_array, sim, label = r"$n = $"+str(n), linestyle = ' ', marker = markers[i_n], ms = 3, color = colors[i_n])
    #axs[0].set_title('Nwks with same percolation percentage: random')
    axs[1].set_yticklabels([])
    axs[1].set(xlabel=r'$f$')

    if i_n == len(nodes_list)-1:
            path_name = "{}_attack{}_n{}.npy".format("infRelSCurve", remove_bool, n)
            path_name = os.path.join("data", "synthetic_data", path_name)
            all_inf = np.load(path_name)
            #print(np.shape(all_inf))
            #inf = all_inf[p_index]
            inf = infiniteTheory.relSCurve(n, p, attack=remove_bool, reverse=False, smooth_end=False)
            axs[1].plot(nodes_array, inf, label = r"${\langle S \rangle}_{N \to \infty}$")
    axs[1].legend()
    # pos = axs[1].get_position()
    # axs[1].set_position([pos.x0, pos.y0, pos.width, pos.height])
    # axs[1].legend(loc='upper right', bbox_to_anchor=(1.35, 1.15))

n = 25
mult_probs = [.05,.3, .7, .9, 1]
nodes_array = np.arange(n)/n
trials = 10
for j in range(len(mult_probs)):
    p_index = int(mult_probs[j]/.01 - 1)
    # axs[1].plot(np.arange(n)/n, finiteTheory.relSCurve(mult_probs[j], n,
    #                                         attack=False, fdict=fvals, pdict=pvals, lcc_method_relS="pmult", executable_path='libs/p-recursion-float128.exe'),
    #                                         label = r"$p=$"+str(mult_probs[j]))

    
    fin = (relSCurve_precalculated(n, mult_probs[j], targeted_removal=remove_bool, simulated=False, finite=True)[:n])

    if mult_probs[j] == .05 or mult_probs[j] ==.1:
        sim = np.zeros(n)
        for k in range(trials):
            print(k)
            sim += finiteTheory.relSCurve(mult_probs[j], n,
                                        attack=remove_bool, fdict=fvals, pdict=pvals, lcc_method_relS="pmult", executable_path='libs/p-recursion-float128.exe')
        sim /= trials
    else:
        all_sim = relSCurve_precalculated(n, mult_probs[j], targeted_removal=remove_bool, simulated=True, finite=False)
        #print("allsim",i,j,all_sim)
        sim = np.zeros(n)
        #print("i",i)
        for k in range(n):
            sim = sim + np.transpose(all_sim[:,k][:n])
        sim = sim / n

    path_name = "{}_attack{}_n{}.npy".format("infRelSCurve", remove_bool, n)
    path_name = os.path.join("data", "synthetic_data", path_name)
    all_inf = np.load(path_name)
    #print(np.shape(all_inf))
    #inf = all_inf[p_index]
    inf = infiniteTheory.relSCurve(n, mult_probs[j], attack=remove_bool, reverse=False, smooth_end=False)
    axs[0].plot(nodes_array, inf, label = r"${\langle S \rangle}_{N \to \infty}$", color = colors[j])

    axs[0].plot(nodes_array, fin, linestyle = '--', color = colors[j])
    axs[0].plot(nodes_array, sim, label = r"$p = $" + str(mult_probs[j]), linestyle = " ", marker = markers[j], ms = 3, color = colors[j])
    #axs[1].set_title('Nwks with same percolation percentage: attack')
    axs[0].set(xlabel=r'$f$', ylabel=r'$\langle S \rangle$')
    pos2 = axs[0].get_position()
    axs[0].set_position([pos2.x0, pos2.y0, pos2.width, pos2.height])
    axs[0].legend(loc='upper right', bbox_to_anchor=(1.30, 1.15))
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
plt.subplots_adjust(left=0.06, right=.99, bottom=.15, top=0.9, wspace=.3)
if remove_bool == False:
    plt.savefig("Fig 2 final")
    plt.savefig("Fig_2_final.pdf")
else:
    plt.savefig("Fig 2 final attack")
    plt.savefig("Fig_2_final_attack.pdf")
