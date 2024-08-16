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
9
fig = plot_graphs(numbers_of_nodes=[5], edge_probabilities=[.1, 0.5, 0.9],
                     graph_types=['ER'], remove_strategies=["attack"], #legend=False,
                     performance='relative LCC', num_trials=300,
                     smooth_end=False, forbidden_values=[], fdict=fvals, 
                     lcc_method_main="pmult", savefig='',
                     simbool=True, executable_path = r".\libs\p-recursion.exe")

colors = ['red','blue','orange','green','purple','cyan','magenta']
n_threshold = .526
nodes_list = [20]
probs_list = [(1/(n_threshold*(x-1))) for x in nodes_list]





plt.subplots_adjust(left=0.08, right=.8, bottom=.15, top=0.9, wspace=.1)
plt.savefig("Fig 2 test")

