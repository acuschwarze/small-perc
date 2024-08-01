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

#fvals = pickle.load(open('data/fvalues.p', 'rb'))
#pvals = pickle.load(open('data/Pvalues.p', 'rb'))



fig = plt.figure(figsize=(8, 8))
perc_rand = pd.read_pickle("percolation_random")
nodes = perc_rand.nodes.values
data = perc_rand.loc[:,"fin theory RLCC"]
n = len(perc_rand.nodes.values)
for j in range(n):
    nodes_array = np.arange(nodes[j], dtype=float)
    nodes_array -= n
    nodes_array *= -1
    nodes_array /= float(n)
    plt.plot(nodes_array, data[j])
plt.savefig("one_perc_graph_random")


# fig = plt.figure(figsize=(8, 8))
# perc_att = pd.read_pickle("percolation_attack")
# nodes = perc_att.nodes.values
# data = perc_att.loc[:,"fin theory RLCC"]
# n = len(perc_att.nodes.values)
# for j in range(n):
#     plt.plot(np.arange((n-nodes[j])/n), data[j])
# plt.savefig("one_perc_graph_attack")