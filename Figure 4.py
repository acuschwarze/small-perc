import sys, pickle
sys.path.insert(0, "libs")

import os, pickle, csv # import packages for file I/O
import time # package to help keep track of calculation time

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
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

fullData = pd.read_csv("fullData.csv")

# making csv of data

# k = len(fullData)

# mse_array = np.zeros((k,4),dtype=object)

# for i in range(k):
#     n = fullData.iloc[i][1]
#     p = fullData.iloc[i][2] / scipy.special.comb(n,2)
#     sim = string2array(fullData.iloc[i][3], sep=" ")
#     fin = string2array(fullData.iloc[i][5], sep=" ")
#     mse = ((fin-sim)**2).mean()

#     mse_array[i][0] = fullData.iloc[i][0]
#     mse_array[i][1] = n
#     mse_array[i][2] = p
#     mse_array[i][3] = mse

# df = pd.DataFrame(mse_array)
# df.to_csv("MSEdata3D.csv")
# df.columns = ["network", "n", "p", "mse"]


# making 3D graph
msedata = pd.read_csv("MSEdata3D.csv")
num_nwks = len(msedata)
nodes_array = np.zeros(num_nwks)
probs_array = np.zeros(num_nwks)
mse_array = np.zeros(num_nwks)

# some weird formatting means you have to add 1 to each index for the n,p,mse
for j in range(num_nwks):
    nodes_array[j] = msedata.iloc[j][2]
    probs_array[j] = msedata.iloc[j][3]
    mse_array[j] = msedata.iloc[j][4]

fig = plt.figure()
 
ax = plt.axes(projection ='3d')
 
#ax.scatter(nodes_array, probs_array, mse_array)
tri = mtri.Triangulation(nodes_array, probs_array)
ax.plot_trisurf(nodes_array, probs_array, mse_array, triangles=tri.triangles, cmap=plt.cm.Spectral)
ax.set_title('MSE over n and p')
plt.show()