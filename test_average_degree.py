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
from scipy.special import comb, binom
from scipy.integrate import simpson
from scipy.signal import argrelextrema
from random import choice

from libs.utils import *
from libs.finiteTheory import *
from libs.performanceMeasures import *
from libs.utils import *





num_trials = 500
remove_nodes = 'attack'
n = 20
p = 0.2
p_new = p

data_array = np.zeros((num_trials, 6, n), dtype=float)
data_array[0] = np.arange(n)



for j in range(num_trials):
    g = sampleNetwork(n, p, graph_type='ER')
    c = averageDegree(g)
    p_new = p
    
    for i in range(n):
            # calculate performance value
            data_array[j, 1, i] = averageDegree(g) # computePerformance(g)
            data_array[j, 2, i] = edgeProbabilityAfterTargetedAttack(n-i, p_new)
            #print((n-i, p_new, averageDegree(g)*(n-i-1)/2, p))
            g2 = sampleNetwork(n-i, p_new, graph_type='ER')
            data_array[j, 3, i] = averageDegree(g2) 
            data_array[j, 4, i] = np.max([d for n, d in g.degree()] )
            if binom(n-i,2):
                data_array[j, 5, i] = expectedMaxDegree(g.number_of_nodes(), g.number_of_edges()/binom(n-i,2)) 
            p_new = data_array[j, 2, i]
            
            if i == n:
                break

            # find a node to remove
            if remove_nodes == 'random':
                # select a random node
                v = choice(list(g.nodes()))
            elif remove_nodes == 'attack':
                # select the most connected node
                v = sorted(g.degree, key=lambda x: x[1], reverse=True)[0][0]
            else:
                raise ValueError('I dont know that mode of removing nodes')
                v = None
            # remove node
            g.remove_node(v)

#print(data_array)


plt.plot(np.mean(data_array[:,1],axis=0), marker='x', label='sim')
plt.plot(np.mean(data_array[:,2]*n*(n-1)/2/n*2,axis=0), marker='x', label='c from pnew')
plt.plot(np.mean(data_array[:,3],axis=0), marker='x', label='sim with pnew')
plt.plot(np.mean(data_array[:,4],axis=0), marker='x', label='max degree')
plt.plot(np.mean(data_array[:,5],axis=0), marker='x', label='expected max degree')
plt.legend()
plt.savefig('test_average_degree.png')
print(np.mean(data_array[:,2]*n*(n-1)/2/n*2,axis=0))