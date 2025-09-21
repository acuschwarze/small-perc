###############################################################################
#
# Code to create tables of MSE values for Petter Holme's real networks, used in Fig 7/8
#
###############################################################################

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


fullData = pd.read_csv("fullData.csv")
k = len(fullData)

# removal type: either "random" or "targeted"
removal = "random"

mse_array = np.zeros((k,4),dtype=object)
for i in range(k):
    # retrieve n and p values
    n = fullData.iloc[i][1]
    p = fullData.iloc[i][2] / scipy.special.comb(n,2)

    # retrieve simulated and finite theory data
    if removal == "random":
        sim = string2array(fullData.iloc[i][3], sep=" ")
        fin = string2array(fullData.iloc[i][5], sep=" ")
    elif removal == "targeted":
        sim = string2array(fullData.iloc[i][4], sep=" ")
        fin = string2array(fullData.iloc[i][6], sep=" ")

    # calculate mean square error
    mse = ((fin-sim)**2).mean()

    mse_array[i][0] = fullData.iloc[i][0]
    mse_array[i][1] = n
    mse_array[i][2] = p
    mse_array[i][3] = mse

df = pd.DataFrame(mse_array)

if removal == "random":
    df.to_csv("MSEdata3D2.csv")
elif removal == "targeted":
    df.to_csv("MSEdata3D2targetedcsv")2

df.columns = ["network", "n", "p", "mse"]