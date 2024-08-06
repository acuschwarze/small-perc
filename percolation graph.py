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

#fvals = pickle.load(open('data/fvalues.p', 'rb'))
#pvals = pickle.load(open('data/Pvalues.p', 'rb'))


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
# print(perc_att)
# #print(perc_att.iloc[5][2])
# print(perc_att.iloc[5][3])
# nodes = perc_att.nodes.values
# data = perc_att.loc[:,"fin theory RLCC"]
# sim = perc_att.loc[:,'simulated RLCC']
# n = len(perc_att.nodes.values)
# for j in range(n):
#     nodes_array = np.arange(nodes[j], dtype=float)/nodes[j]
#     #plt.plot(nodes_array, data[j], color = colors[j])
#     plt.plot(nodes_array,sim[j],'o',color = colors[j])
# p = 1/(.2*(100-1))
# # plt.plot(nodes_array, infiniteTheory.relSCurve(100, p,
# #                             attack=True, smooth_end=False), label = "inf theory")
# plt.xlabel("percent nodes removed")
# plt.ylabel("relative LCC")
# plt.title("relative LCC over nodes removed targeted")
# plt.savefig("one_perc_graph_attack")

# p=1/(.2*(100-1))
# print(finiteTheory.relSCurve(p,100,
#                                 attack=True, fdict=fvals,pdict=pvals,lcc_method_relS="pmult",executable_path = "C:\\Users\\jj\\Downloads\\GitHub\\small-perc\\p-recursion.exe"))


# y = [9.93882333e-01,  9.92612841e-01,  9.91094892e-01,  9.89279175e-01,
#   9.87112169e-01,  9.84530057e-01,  9.81452867e-01,  9.77788086e-01,
#   9.73431430e-01,  9.68245848e-01,  9.62080548e-01,  9.54746296e-01,
#   9.46016599e-01,  9.35614558e-01,  9.23207006e-01,  9.08372073e-01,
#   8.90576858e-01,  8.69121838e-01,  8.43061571e-01,  8.11099479e-01,
#   7.71557204e-01,  7.22618414e-01,  6.63062232e-01,  5.93356704e-01,
#   5.16446365e-01,  4.37354019e-01,  3.62006036e-01,  2.95837021e-01,
#   1.90388614e-01, -4.16714791e-02, -2.50526178e+00, -5.46617285e+01,
#  -1.29389143e+01, -2.63785777e+02,  8.01131395e+02, -1.23268822e+03,
#  -2.44352034e+03,  4.72274114e+03, -2.13964728e+03, -2.10601973e+04,
#  -4.18066324e+03,  7.56484124e+00, -9.46290804e+03,  8.59685962e+03,
#   1.78571429e-02,  1.81818182e-02,  1.85185185e-02,  1.88679245e-02,
#   1.92307692e-02,  1.96078431e-02,  2.00000000e-02,  2.04081633e-02,
#   2.08333333e-02,  2.12765957e-02,  2.17391304e-02,  2.22222222e-02,
#   2.27272727e-02,  2.32558140e-02,  2.38095238e-02,  2.43902439e-02,
#   2.50000000e-02,  2.56410256e-02,  2.63157895e-02,  2.70270270e-02,
#   2.77777778e-02,  2.85714286e-02,  2.94117647e-02,  3.03030303e-02,
#   3.12500000e-02,  3.22580645e-02,  3.33333333e-02,  3.44827586e-02,
#   3.57142857e-02,  3.70370370e-02,  3.84615385e-02,  4.00000000e-02,
#   4.16666667e-02,  4.34782609e-02,  4.54545455e-02,  4.76190476e-02,
#   5.00000000e-02,  5.26315789e-02,  5.55555556e-02,  5.88235294e-02,
#   6.25000000e-02,  6.66666667e-02,  7.14285714e-02,  7.69230769e-02,
#   8.33333333e-02,  9.09090909e-02,  1.00000000e-01,  1.11111111e-01,
#   1.25000000e-01,  1.42857143e-01,  1.66666667e-01,  2.00000000e-01,
#   2.50000000e-01,  3.33333333e-01,  5.00000000e-01,  1.00000000e+00]

# x = np.arange(100)/100
# plt.plot(x,y)
# plt.show()

# p=1/(.4*9)
# print(finiteTheory.relSCurve(p,10,
#                                  attack=True, fdict=fvals,pdict=pvals,lcc_method_relS="pmult",executable_path = "C:\\Users\\jj\\Downloads\\GitHub\\small-perc\\p-recursion.exe"))
#one_perc_thresh_table(threshold=.4, nodes=[10, 15, 25, 50, 75, 100], removal=["attack"])

root = r'C:\Users\jj\Downloads\GitHub\small-perc\nwks small perc'
pattern = "*.adj"
pattern2 = "*.arc"
nwks_list2 = []

for path, subdirs, files in os.walk(root):
    for name in files:
        if fnmatch(name, pattern):
            # print(os.path.join(path, name))
            nwks_list2.append(os.path.join(path, name))
        elif fnmatch(name, pattern2):
            nwks_list2.append(os.path.join(path, name))

df = pd.DataFrame(nwks_list2)
df.to_csv("nwks_list2")
