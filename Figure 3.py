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


def add_colorbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar



total_n = 30
total_p = 100
nodes_array = np.arange(1,total_n+1)
probs_array = np.linspace(.01,1,total_p)


def fig_3(total_n,total_p,remove_bool):
    heatmap0 = np.zeros((total_p,total_n))
    heatmap1 = np.zeros((total_p,total_n))
    heatmap2 = np.zeros((total_p,total_n))
    heatmap3 = np.zeros((total_p,total_n))

    nodes_array = np.arange(1,total_n+1)
    probs_array = np.linspace(.01,1,total_p)

    for i in range(1,total_n+1):
        for j in range(total_p):
            all_sim = relSCurve_precalculated(i, probs_array[j], targeted_removal=remove_bool, simulated=True, finite=False)
            sim = np.zeros(i)
            for k in range(i):
                sim = sim + np.transpose(all_sim[:,k][:i])
            sim = sim / i

            fin = (relSCurve_precalculated(i, probs_array[j], targeted_removal=remove_bool, simulated=False, finite=True)[:i])
            
            inf = infiniteTheory.relSCurve(i, probs_array[j], attack=remove_bool, smooth_end=False)

            if i==1:
                heatmap0[j][i-1] = 1
            else:
                heatmap0[j][i-1] = scipy.integrate.simpson(sim, dx=1 / (i - 1))

            heatmap1[j][i-1] = ((fin-sim)**2).mean()

            heatmap2[j][i-1] = ((inf-sim)**2).mean()

            inf_mse = ((inf-sim)**2).mean()
            fin_mse = ((fin-sim)**2).mean()
            heatmap3[j][i-1] = inf_mse-fin_mse

    heatmap0 = heatmap0.tolist()
    heatmap1 = heatmap1.tolist()
    heatmap2 = heatmap2.tolist()
    heatmap3 = heatmap3.tolist()

    xnodes, yprobs = np.meshgrid(nodes_array, probs_array)

    fig , ( (ax1,ax2) , (ax3,ax4)) = plt.subplots(2, 2,sharex = True,sharey=True)
    z1_plot = ax1.pcolormesh(xnodes, yprobs, heatmap0, cmap = "Reds")
    z2_plot = ax2.pcolormesh(xnodes, yprobs, heatmap1, vmax=0.05)
    z3_plot = ax3.pcolormesh(xnodes, yprobs, heatmap3, cmap = "Blues")
    z4_plot = ax4.pcolormesh(xnodes, yprobs, heatmap2, vmax=0.2)
    add_colorbar(z1_plot)
    add_colorbar(z2_plot)
    add_colorbar(z3_plot)
    add_colorbar(z4_plot)
    ax1.set(ylabel='p')
    #ax2.set(xlabel='nodes')
    ax3.set(xlabel=r'N',ylabel=r'p')
    ax4.set(xlabel=r'N')
    ax1.set_title("AUC")
    ax2.set_title("MSE Finite")
    ax3.set_title("MSE Difference")
    ax4.set_title('MSE Infinite')

    plt.savefig("Fig 3 Final")

fig_3(35,100,False)


# #auc

# heatmap0 = np.zeros((total_p,total_n))
# for i in range(1,total_n+1):
#     for j in range(total_p):
#         all_sim = relSCurve_precalculated(i, probs_array[j], targeted_removal=False, simulated=True, finite=False)
#         #print("allsim",i,j,all_sim)
#         sim = np.zeros(i)
#         #print("i",i)
#         for k in range(i):
#             sim = sim + np.transpose(all_sim[:,k][:i])
#         sim = sim / i
#         if i==1:
#             heatmap0[j][i-1] = 1
#         else:
#             heatmap0[j][i-1] = scipy.integrate.simpson(sim, dx=1 / (i - 1))

# heatmap0 = heatmap0.tolist()
# xnodes, yprobs = np.meshgrid(nodes_array, probs_array)

# # plt.pcolormesh(xnodes, yprobs, heatmap0)
# # plt.xlabel("nodes")
# # plt.ylabel("probability")
# # plt.title("heatmap of AUC MSE")
# # plt.colorbar()
# # plt.savefig("mse auc")


# # finite
# heatmap1 = np.zeros((total_p,total_n))
# for i in range(1,total_n+1):
#     for j in range(total_p):
#         all_sim = relSCurve_precalculated(i, probs_array[j], targeted_removal=False, simulated=True, finite=False)
#         #print("allsim",i,j,all_sim)
#         sim = np.zeros(i)
#         #print("i",i)
#         for k in range(i):
#             sim = sim + np.transpose(all_sim[:,k][:i])
#         sim = sim / i
#         #print("sim",sim)
#         fin = (relSCurve_precalculated(i, probs_array[j], targeted_removal=False, simulated=False, finite=True)[:i])
#         #print("fin",fin)
#         heatmap1[j][i-1] = ((fin-sim)**2).mean()
# maxf = np.max(heatmap1)
# print("max fin", maxf)
# medf = np.median(heatmap1)
# print("median fin", medf)


# heatmap1 = heatmap1.tolist()
# # print(heatmap1)
# # xnodes, yprobs = np.meshgrid(nodes_array, probs_array)

# # plt.pcolormesh(xnodes, yprobs, heatmap1)
# # plt.xlabel("nodes")
# # plt.ylabel("probability")
# # plt.title("heatmap of AUC MSE")
# # plt.colorbar()
# # plt.savefig("mse finite")


# # #infinite
# heatmap2 = np.zeros((total_p,total_n))
# targeted_removal = False
# for i in range(1,total_n+1):
#     for j in range(total_p):
#         all_sim = relSCurve_precalculated(i, probs_array[j], targeted_removal=False, simulated=True, finite=False)
#         #print("allsim",i,j,all_sim)
#         sim = np.zeros(i)
#         #print("i",i)
#         for k in range(i):
#             sim = sim + np.transpose(all_sim[:,k][:i])
#         sim = sim / i
#         #print(np.shape(sim))
#         path_name = "{}_attack{}_n{}.npy".format("infRelSCurve", targeted_removal, i)
#         path_name = os.path.join("data", "synthetic_data", path_name)
#         all_inf = np.load(path_name)
#         #print(np.shape(all_inf))
#         inf = all_inf[j]
#         #print("i", i, "j", j, "inf", inf)
#         heatmap2[j][i-1] = ((inf-sim)**2).mean()

# heatmap2 = heatmap2.tolist()
# maxi = np.max(heatmap2)
# print("max inf", maxi)
# medi = np.median(heatmap2)
# print("median inf", medi)
# #xnodes, yprobs = np.meshgrid(nodes_array, probs_array)

# # plt.pcolormesh(xnodes, yprobs, heatmap2)
# # plt.xlabel("nodes")
# # plt.ylabel("probability")
# # plt.title("heatmap of AUC MSE")
# # plt.colorbar()
# # plt.savefig("mse infinite")


# #difference
# heatmap3 = np.zeros((total_p,total_n))
# targeted_removal = False
# for i in range(1,total_n+1):
#     for j in range(total_p):
#         all_sim = relSCurve_precalculated(i, probs_array[j], targeted_removal=False, simulated=True, finite=False)
#         #print("allsim",i,j,all_sim)
#         sim = np.zeros(i)
#         #print("i",i)
#         for k in range(i):
#             sim = sim + np.transpose(all_sim[:,k][:i])
#         sim = sim / i
#         #print(np.shape(sim))
#         path_name = "{}_attack{}_n{}.npy".format("infRelSCurve", targeted_removal, i)
#         path_name = os.path.join("data", "synthetic_data", path_name)
#         all_inf = np.load(path_name)
#         #print(np.shape(all_inf))
#         inf = all_inf[j]
#         #print("i", i, "j", j, "inf", inf)
#         fin = (relSCurve_precalculated(i, probs_array[j], targeted_removal=False, simulated=False, finite=True)[:i])
#         inf_mse = ((inf-sim)**2).mean()
#         fin_mse = ((fin-sim)**2).mean()
#         heatmap3[j][i-1] = inf_mse-fin_mse

# heatmap3 = heatmap3.tolist()
# #xnodes, yprobs = np.meshgrid(nodes_array, probs_array)

# # plt.pcolormesh(xnodes, yprobs, heatmap3)
# # plt.xlabel("nodes")
# # plt.ylabel("probability")
# # plt.title("heatmap of AUC MSE")
# # plt.colorbar()
# # plt.savefig("mse difference")

# def add_colorbar(mappable):
#     from mpl_toolkits.axes_grid1 import make_axes_locatable
#     import matplotlib.pyplot as plt
#     last_axes = plt.gca()
#     ax = mappable.axes
#     fig = ax.figure
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="5%", pad=0.05)
#     cbar = fig.colorbar(mappable, cax=cax)
#     plt.sca(last_axes)
#     return cbar


# fig , ( (ax1,ax2) , (ax3,ax4)) = plt.subplots(2, 2,sharex = True,sharey=True)
# z1_plot = ax1.pcolormesh(xnodes, yprobs, heatmap0, cmap = "Reds")
# z2_plot = ax2.pcolormesh(xnodes, yprobs, heatmap1, vmax=0.05)
# z3_plot = ax3.pcolormesh(xnodes, yprobs, heatmap3, cmap = "Blues")
# z4_plot = ax4.pcolormesh(xnodes, yprobs, heatmap2, vmax=0.2)
# add_colorbar(z1_plot)
# add_colorbar(z2_plot)
# add_colorbar(z3_plot)
# add_colorbar(z4_plot)
# ax1.set(ylabel='p')
# #ax2.set(xlabel='nodes')
# ax3.set(xlabel='nodes',ylabel='p')
# ax4.set(xlabel='nodes')
# ax1.set_title("AUC")
# ax2.set_title("MSE Finite")
# ax3.set_title("MSE Difference")
# ax4.set_title('MSE Infinite')

# plt.savefig("Fig 3 Final")

# # fig, axs = plt.subplots(2, 2)
# # axs[0, 0].pcolormesh(xnodes, yprobs, heatmap0)
# # axs[0, 0].set_title('AUC')
# # axs[0, 0].set(xlabel='nodes', ylabel='p')
# # axs[0, 1].pcolormesh(xnodes, yprobs, heatmap1, vmax = 0.05)
# # axs[0, 1].set_title('MSE Finite')
# # axs[0, 1].set(xlabel='nodes', ylabel='p')
# # axs[1, 0].pcolormesh(xnodes, yprobs, heatmap3)
# # axs[1, 0].set_title('MSE Difference')
# # axs[1, 0].set(xlabel='nodes', ylabel='p')
# # axs[1, 1].pcolormesh(xnodes, yprobs, heatmap2, vmax = .2)
# # axs[1, 1].set_title('MSE Infinite')
# # axs[1, 1].set(xlabel='nodes', ylabel='p')





# #im = axs[1, 1].imshow(heatmap2)
# #plt.colorbar(im, ax=axs[1, 1])

# # for ax in axs.flat:
#     # ax.set(xlabel='nodes', ylabel='MSE')

# # Hide x labels and tick labels for top plots and y ticks for right plots.
# # for ax in axs.flat:
# #     ax.label_outer()