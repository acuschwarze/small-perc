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

fullData = pd.read_csv("fullData.csv")
k = len(fullData)

mse_array = np.zeros((k,4),dtype=object)

for i in range(k):
    n = fullData.iloc[i][1]
    p = fullData.iloc[i][2] / scipy.special.comb(n,2)
    sim = string2array(fullData.iloc[i][3], sep=" ")
    fin = string2array(fullData.iloc[i][5], sep=" ")
    mse = ((fin-sim)**2).mean()

    mse_array[i][0] = fullData.iloc[i][0]
    mse_array[i][1] = n
    mse_array[i][2] = p
    mse_array[i][3] = mse

df = pd.DataFrame(mse_array)
df.to_csv("MSEdata3D.csv")
df.columns = ["network", "n", "p", "mse"]