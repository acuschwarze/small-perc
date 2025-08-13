###############################################################################
#
# Code to create Figure 1 of paper
#
###############################################################################

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
from scipy.special import comb, lambertw
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

# open precalculated values for recursion as part of finite theory S_rec (optional, not used in Fig 1)
fvals = pickle.load(open('data/fvalues.p', 'rb'))
pvals = pickle.load(open('data/Pvalues.p', 'rb'))


# lambert function for infinite theory
def myLambertW(x, k=0, tol=1E-20):
    '''Lambert-W function with interpolation close to the jump point of its 
    zero-th branch. (Using the scipy implementation sometimes does not return
    a number if evaluated too close to the jump point.)

    Parameters
    ----------
    x : float
       Argument of the Lambert-W function.

    k : int (default=0)
       Branch of the Lambert-W function.

    Returns
    -------
    lw : float
       Value of the Lambert-W function (with interpolation near jump point)
    '''

    if np.abs(x + 1 / np.exp(1)) < tol:
        # if input is close to percolation threshold, set output to -1.0
        lw = -1.0
    else:
        lw = lambertw(x, k=k)

    return lw

# parameters for node number (n) and p values (here 0 to .5)
n = 10
probs = np.arange(51)/100

# create curves for finite theory S_rec and simulated data
fin = np.zeros(len(probs))
inf = np.zeros(len(probs))
sim = np.zeros(len(probs))

# finite theory S_rec (optional)
for i in range(len(probs)):
    fin[i] = calculate_S(probs[i], n, fdict=fvals, pdict=pvals,lcc_method = "pmult", executable_path='libs/p-recursion-float128.exe')/n


# infinite theory S_inf
for i in range(len(probs)):
        c = 2 * probs[i] * comb(n, 2) / n

        # compute value of S from percolation theory for infinite networks
        if c == 1 and n==2:
            inf[i] = 2/n
        
        elif c > 0:
            inf[i] = 1 + np.real(
                myLambertW((-c * np.exp(-c)), k=0, tol=1e-8) / c)

        else:
            inf[i] = 0


# simulated data with 3 standard error bars
std_table = np.zeros(len(probs))

for i in range(len(probs)):
    std = np.zeros(100)
    lcc = 0
    for j in range(100):
        G = nx.erdos_renyi_graph(n,probs[i])
        std[j] = len(max(nx.connected_components(G), key=len))/n
        lcc += len(max(nx.connected_components(G), key=len))/n
    std_value = np.std(std)
    std_table[i] = std_value / 10 * 3 # 3 standard errors

    lcc /= 100
    sim[i] = lcc


# plot figure
fig, axs = plt.subplots(1,1, figsize = [5,3.5])
# plt.plot(probs, fin, label = r'$\langle S \rangle$', linestyle = "--", color = "blue") # optional S_rec
plt.errorbar(x=probs, y=sim, yerr = std_table, marker = 'o', markersize=2.5, label = r"$\widebar{S}$", lw=1, color = "red")
plt.plot(probs, inf, label = r"${S}_{\infty}$", color = "black")
plt.xlabel("edge probability " + r"$p$")
plt.ylabel("rel. LCC size")
plt.legend()
handles, labels = plt.gca().get_legend_handles_labels()
order = [1,0]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
plt.subplots_adjust(left=0.12, right=.99, bottom=.15, top=0.99, wspace=0)
plt.savefig("Figure_1.pdf")
