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
from matplotlib.gridspec import GridSpec

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



def add_colorbar_neg(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    newticks1 = cbar.ax.get_yticklabels()
    newticks1 = [label.get_text() for label in newticks1]
    newticks1 = [a.replace('−', '-') for a in newticks1]
    #newticks1 = [int(a) for a in newticks1 if "." not in a]
    #newticks1 = [float(a) for a in newticks1 if type(a) != int]
    newticks2 = [r'$10^{{{}}}$'.format(x) for x in newticks1]
    cbar.ax.set_yticklabels(newticks2) 
    plt.sca(last_axes)
    return cbar

def add_colorbar_norm(mappable):
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

max_n = 30
total_p = 100
nodes_array = np.arange(2,max_n+1)
probs_array = np.linspace(.01,1,total_p)



def fig_3(max_n,total_p):
    nodes_array = np.arange(2,max_n+1)
    # probs_array = np.linspace(.01,1,total_p)

    heatmap_rauc = np.zeros((total_p,len(nodes_array)))
    heatmap_rfin = np.zeros((total_p,len(nodes_array)))
    heatmap_rinf = np.zeros((total_p,len(nodes_array)))

    heatmap_tauc = np.zeros((total_p,len(nodes_array)))
    heatmap_tfin = np.zeros((total_p,len(nodes_array)))
    heatmap_tinf = np.zeros((total_p,len(nodes_array)))

    badcount = 0
    for i in range(2,max_n+1):
        for j in range(total_p):
            r_all_sim = relSCurve_precalculated(i, probs_array[j], targeted_removal=False, simulated=True, finite=False)
            r_sim = np.zeros(i)
            for k in range(i):
                r_sim = r_sim + np.transpose(r_all_sim[:,k][:i])
            r_sim = r_sim / i

            r_fin = (relSCurve_precalculated(i, probs_array[j], targeted_removal=False, simulated=False, finite=True)[:i])
            
            r_inf = infiniteTheory.relSCurve(i, probs_array[j], attack=False, smooth_end=False)


            t_all_sim = relSCurve_precalculated(i, probs_array[j], targeted_removal=True, simulated=True, finite=False)
            t_sim = np.zeros(i)
            for k in range(i):
                t_sim = t_sim + np.transpose(t_all_sim[:,k][:i])
            t_sim = t_sim / i

            t_fin = (relSCurve_precalculated(i, probs_array[j], targeted_removal=True, simulated=False, finite=True)[:i])
            
            t_inf = infiniteTheory.relSCurve(i, probs_array[j], attack=True, smooth_end=False)

            if i==1:
                heatmap_rauc[j][i-2] = 1
                heatmap_tauc[j][i-2] = 1
            else:
                heatmap_rauc[j][i-2] = scipy.integrate.simpson(r_sim, dx=1 / (i - 1))
                heatmap_tauc[j][i-2] = scipy.integrate.simpson(t_sim, dx=1 / (i - 1))
            
            # heatmap_rfin[j][i-2] = ((r_fin-r_sim)**2).mean() * i
            # heatmap_rinf[j][i-2] = ((r_inf-r_sim)**2).mean() * i
            # heatmap_tfin[j][i-2] = ((t_fin-t_sim)**2).mean() * i
            # heatmap_tinf[j][i-2] = ((t_inf-t_sim)**2).mean() * i

            if ((r_fin-r_sim)**2).mean() == 0:
                heatmap_rfin[j][i-2] = -6
            else:
                heatmap_rfin[j][i-2] = np.log10(((r_fin-r_sim)**2).mean())

            if ((r_inf-r_sim)**2).mean() == 0:
                heatmap_rinf[j][i-2] = -2
            else:
                heatmap_rinf[j][i-2] = np.log10(((r_inf-r_sim)**2).mean())

            if ((t_fin-t_sim)**2).mean() == 0:
                heatmap_tfin[j][i-2] = -7 # -6
            elif ((t_fin-t_sim)**2).mean() <= 10**(-7):
                heatmap_tfin[j][i-2] == -7
                badcount+=1
            else:
                heatmap_tfin[j][i-2] = np.log10(((t_fin-t_sim)**2).mean())

            if ((t_inf-t_sim)**2).mean() == 0:
                heatmap_tinf[j][i-2] = -2
            else:
                heatmap_tinf[j][i-2] = np.log10(((t_inf-t_sim)**2).mean())

    print("badcount",badcount)
    heatmap_rauc = heatmap_rauc.tolist()
    heatmap_rfin = heatmap_rfin.tolist()
    heatmap_rinf = heatmap_rinf.tolist()
    heatmap_tauc = heatmap_tauc.tolist()
    heatmap_tfin = heatmap_tfin.tolist()
    heatmap_tinf = heatmap_tinf.tolist()


    hist_rfin = np.ravel(heatmap_rfin)
    hist_rinf = np.ravel(heatmap_rinf)
    hist_tfin = np.ravel(heatmap_tfin)
    hist_tinf = np.ravel(heatmap_tinf)
    
    xnodes, yprobs = np.meshgrid(nodes_array, probs_array)

    fig, ((ax1,ax2,ax3,ax4,ax5),(ax6,ax7,ax8,ax9,ax10)) = plt.subplots(nrows=2, ncols=5, figsize=[13,6],
                         gridspec_kw={"width_ratios" : [1,1,1,.01,1], "wspace":.3})
 
    plt.subplots_adjust(left=0.05, right=.99, bottom=.1, top=0.9)

    # replacing 'gnuplot2' with just the last 75% of 'gnuplot2':
    from matplotlib import cm
    cmap = cm.get_cmap('gnuplot2')
    new_cmap = cm.colors.ListedColormap(cmap(np.linspace(0.3, 1.0, 256)))
    reversed = new_cmap.reversed()

    z1_plot = ax1.pcolormesh(xnodes, yprobs, heatmap_rauc)
    z2_plot = ax2.pcolormesh(xnodes, yprobs, heatmap_rfin, cmap = reversed, vmin=-4,vmax= -.3)#vmin=-5) #, vmax=0.01)
    z3_plot = ax3.pcolormesh(xnodes, yprobs, heatmap_rinf, cmap = reversed, vmin = -4,vmax= -.3)
    z5_plot = ax5.hist(hist_rfin, density=True, label = r"${\langle S \rangle$}_{}", alpha=0.65)
    ax5.hist(hist_rinf, density=True, label = r"${\langle S \rangle}_{N \to \infty}$", alpha=0.65)
    ax5.legend(prop={'size': 10})

    z6_plot = ax6.pcolormesh(xnodes, yprobs, heatmap_tauc)
    z7_plot = ax7.pcolormesh(xnodes, yprobs, heatmap_tfin, cmap = reversed, vmin = -4,vmax= -.3)
    z8_plot = ax8.pcolormesh(xnodes, yprobs, heatmap_tinf, cmap = reversed, vmin = -4,vmax= -.3)
    z10_plot = ax10.hist(hist_tfin, density=True, label = r"${\langle S \rangle$}_{}", alpha=0.65)
    ax10.hist(hist_tinf, density=True, label = r"${\langle S \rangle}_{N \to \infty}$", alpha=0.65)
    ax10.legend(prop={'size': 10})
    ax5.set_ylim([0,1])
    ax10.set_xlim([-7,0])
    ax10.set_ylim([0,1])
    
    ax2.set_yticklabels([])
    ax3.set_yticklabels([])
    ax4.axis("off")
    ax7.set_yticklabels([])
    ax8.set_yticklabels([])
    ax9.axis("off")
    #ax10.set_yticklabels([])

    ax5.set_xticklabels([])
    ax1.set_xticklabels([])
    ax2.set_xticklabels([])
    ax3.set_xticklabels([])

    print(ax10.get_xticklabels())
    labels=ax10.get_xticklabels()
    newticks1 = [label.get_text() for label in ax10.get_xticklabels()]
    newticks1 = [a.replace('−', '-') for a in newticks1]
    newticks1 = [int(a) for a in newticks1]
    #newticks1 = [float(label.replace('−', '-').get_text()) for label in ax10.get_xticklabels()]
    newticks2 = [r'$10^{{{}}}$'.format(x) for x in newticks1]
    ax10.set_xticklabels(newticks2)

    add_colorbar_norm(z1_plot)
    add_colorbar_neg(z2_plot)
    add_colorbar_neg(z3_plot)
    add_colorbar_norm(z6_plot)
    add_colorbar_neg(z7_plot)
    add_colorbar_neg(z8_plot)

    ax1.set(ylabel=r'$p$')
    ax5.set(ylabel=r'$frequency$')
    ax6.set(xlabel=r'$N$',ylabel=r'$p$')
    ax7.set(xlabel=r'$N$')
    ax8.set(xlabel=r'$N$')
    ax10.set(xlabel=r'log scale of $(MSE)$',ylabel=r'$frequency$')

    ax1.set_title(r"$\widebar{S} \, AUC$")
    ax2.set_title(r"${\langle S \rangle} \, \log(MSE)$")
    ax3.set_title(r'${\langle S \rangle}_{N \to \infty} \, \log(MSE)$')
    ax5.set_title("Histogram of " + r"log scale of $(MSE)$")
    # ax6.set_title(r"$AUC$")
    # ax7.set_title(r"${\langle S \rangle} MSE$")
    # ax8.set_title(r"${\langle S \rangle}_{N \to \infty} MSE$")
    # ax10.set_title("MSE Histogram")

    plt.savefig("Fig_3_Final.pdf")

fig_3(50,100)


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



# max_n = 30
# total_p = 100
# nodes_array = np.arange(2,max_n+1)
# probs_array = np.linspace(.01,1,total_p)


# def fig_3(max_n,total_p,remove_bool):
#     nodes_array = np.arange(2,max_n+1)
#     probs_array = np.linspace(.01,1,total_p)

#     heatmap0 = np.zeros((total_p,len(nodes_array)))
#     heatmap1 = np.zeros((total_p,len(nodes_array)))
#     heatmap2 = np.zeros((total_p,len(nodes_array)))
#     heatmap3 = np.zeros((total_p,len(nodes_array)))

#     for i in range(2,max_n+1):
#         for j in range(total_p):
#             all_sim = relSCurve_precalculated(i, probs_array[j], targeted_removal=remove_bool, simulated=True, finite=False)
#             sim = np.zeros(i)
#             for k in range(i):
#                 sim = sim + np.transpose(all_sim[:,k][:i])
#             sim = sim / i

#             fin = (relSCurve_precalculated(i, probs_array[j], targeted_removal=remove_bool, simulated=False, finite=True)[:i])
            
#             inf = infiniteTheory.relSCurve(i, probs_array[j], attack=remove_bool, smooth_end=False)

#             if i==1:
#                 heatmap0[j][i-2] = 1
#             else:
#                 heatmap0[j][i-2] = scipy.integrate.simpson(sim, dx=1 / (i - 1))
            
#             # heatmap1[j][i-2] = ((fin-sim)**2).mean()
#             # heatmap2[j][i-2] = ((inf-sim)**2).mean()

#             if ((fin-sim)**2).mean() == 0:
#                 heatmap1[j][i-2] = -6
#             else:
#                 heatmap1[j][i-2] = np.log10(((fin-sim)**2).mean())

#             if ((inf-sim)**2).mean() == 0:
#                 heatmap2[j][i-2] = -2
#             else:
#                 heatmap2[j][i-2] = np.log10(((inf-sim)**2).mean())

#             inf_mse = ((inf-sim)**2).mean()
#             #if inf_mse == 1:
#                 #print("n",i,"p",probs_array[j])
#             fin_mse = ((fin-sim)**2).mean()
#             heatmap3[j][i-2] = (inf_mse-fin_mse)

#     heatmap0 = heatmap0.tolist()
#     heatmap1 = heatmap1.tolist()
#     heatmap2 = heatmap2.tolist()
#     heatmap3 = heatmap3.tolist()
#     #print(heatmap1)
#     #print(heatmap2)

#     hist1 = np.ravel(heatmap1)
#     hist2 = np.ravel(heatmap2)
#     xnodes, yprobs = np.meshgrid(nodes_array, probs_array)

#     # total = len(hist1)+len(hist2)
#     # plt.hist(hist1,density=True,stacked=True)
#     # plt.hist(hist2,density=True,stacked=True)
#     # plt.ylim([0, 1])

#     fig , ( (ax1,ax2) , (ax3,ax4)) = plt.subplots(2, 2)
#     z1_plot = ax1.pcolormesh(xnodes, yprobs, heatmap0, cmap = "Reds")
#     z2_plot = ax2.pcolormesh(xnodes, yprobs, heatmap1)#vmin=-5) #, vmax=0.01)
#     z4_plot = ax4.pcolormesh(xnodes, yprobs, heatmap2)
#     #z3_plot = ax3.pcolormesh(xnodes, yprobs, heatmap3, cmap = "Blues")
#     z3_plot = ax3.hist(hist1,density=True,label = "fin mse",alpha=0.65)
#     ax3.hist(hist2,density=True, label = "inf mse",alpha=0.65)
#     ax3.legend(prop={'size': 10})
#     ax1.set_xlim([2,max_n])
#     ax2.set_xlim([2,max_n])
#     ax4.set_xlim([2,max_n])
#     ax3.set_xlim([min(np.min(heatmap1),np.min(heatmap2)),max(np.max(heatmap1),np.max(heatmap2))])
#     ax1.set_ylim([0,1])
#     ax2.set_ylim([0,1])
#     ax4.set_ylim([0,1])
#     ax3.set_ylim([0,1])
#     add_colorbar(z1_plot)
#     add_colorbar(z2_plot)
#     #add_colorbar(z3_plot)
#     add_colorbar(z4_plot)
#     ax1.set(ylabel=r'$p$')
#     ax3.set(xlabel=r'$MSE$',ylabel=r'$ \% data points$')
#     ax4.set(xlabel=r'$N$',ylabel=r'$p$')
#     ax1.set_title("AUC")
#     ax2.set_title("MSE Finite")
#     ax3.set_title("Histogram")
#     ax4.set_title('MSE Infinite')

#     plt.savefig("Fig 3 Final")

# fig_3(30,100,False)


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