import sys, pickle
sys.path.insert(0, "libs")
from visualizations import *
from utils import *
from robustnessSimulations import *
from performanceMeasures import *
from infiniteTheory import *
from finiteTheory import *
import numpy as np
import scipy
import scipy.stats as sst
import networkx as nx
from random import choice
from scipy.special import comb
import matplotlib.pyplot as plt

def bump(n,i,p):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    for ip in range(len(p)):
        x_1 = np.zeros(i)
        y_1 = np.zeros(i)
        simx = np.zeros(i)
        simy = np.zeros(i)
        G = nx.erdos_renyi_graph(n, p[ip])
        for ii in range(i):
            x_1[ii] = ii
            y_1[ii] = raw_f(p[ip],ii,n)
            simx[ii] = ii
            cc = [len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]
            sizei = 0
            for sizes in cc:
                if sizes == ii:
                    sizei+=1
            simy[ii] = sizei/len(cc)
        ax1.plot(x_1,y_1)
        ax2.plot(simx,simy)

bump(20,5,[.1,.3])
