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

# random removals
n=50
nodes = np.arange(n)/n
p=.08
p_index = int(p/.01 - 1)

for i in range(1,10):

    for j in range(1,i):

        n = 100
        nodes = np.arange(n)

        data = completeRCData(numbers_of_nodes=[n], edge_probabilities=[p],
            num_trials=j*100, performance='relative LCC',
            graph_types=['ER'], remove_strategies=["random"])[0][0][0][0][1:]

        print(data.shape)
        mean = np.mean(data, axis=0)
        ste = np.std(data, axis=0)/np.sqrt(j*100)
        plt.plot(mean)
        plt.errorbar(x=nodes, y=mean, yerr = ste, label = "{} trials".format(j*100))
    
    plt.savefig('test_ste{}.png'.format(i*100))
