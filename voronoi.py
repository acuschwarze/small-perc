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


def accel_asc(n):
    a = [0 for i in range(n + 1)]
    k = 1
    y = n - 1
    while k != 0:
        x = a[k - 1] + 1
        k -= 1
        while 2 * x <= y:
            a[k] = x
            y -= x
            k += 1
            l = k + 1
            while x <= y:
                a[k] = x
                a[l] = y
                yield a[:k + 2]
                x += 1
                y -= 1
                a[k] = x + y
            y = x + y - 1
            yield a[:k + 1]

accel_asc(10)

from scipy.special import comb, lambertw
import numpy as np


for c in [0.9, 1, 1.1]:
    print(1+lambertw(-c * np.exp(-c),k=0))
    print(1+np.real(lambertw(-c * np.exp(-c),k=0)))


import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.cm as cm

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


log_array = np.zeros(len(mse_array))
for i in range(len(mse_array)):
    if np.log(mse_array[i]) <= -4:
        log_array[i] = -4
    elif np.log(mse_array[i]) >= 10**.5:
        log_array[i] = 10**.5

    elif mse_array[i] != 0:
        log_array[i] = np.log(mse_array[i])
    else:
        log_array[i] = -4

mse_array = log_array


points = np.column_stack((nodes_array/100, probs_array))
ring = np.array([[np.cos(x), np.sin(x)] for x in np.linspace(0,7)])*10
ring_z = np.array([1 for x in np.linspace(0,7)])
points = np.concatenate([points, ring])
mse_array = np.concatenate([mse_array, ring_z])

# Generate Voronoi diagram
vor = Voronoi(points)

# Normalize z values to get a colormap
norm = plt.Normalize(vmin=min(mse_array), vmax=.5)
cmap = cm.gnuplot2 #cm.viridis

# Create a plot
fig, ax = plt.subplots()

# Plot Voronoi diagram with cells colored based on z values
voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='grey', line_width=1, line_alpha=0.6, point_size=1)

# Color each Voronoi region
for region_index, region in enumerate(vor.regions):
    if not -1 in region and len(region) > 0:
        polygon = [vor.vertices[i] for i in region]
        color = cmap(norm(mse_array[region_index]))
        ax.fill(*zip(*polygon), color=color)

# Plot the original points
#ax.plot(points[:, 0], points[:, 1], 'ko')



# Add a color bar to show the mapping of z values to colors
sm = plt.cm.ScalarMappable(cmap="gnuplot2", norm=norm)
sm.set_array([])
plt.colorbar(sm, ax=ax, label='Z value')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Voronoi Diagram Colored by Z Values')
plt.xlim([0,1])
plt.ylim([0,1])
plt.savefig("voronoi.pdf")
plt.show()

from scipy.spatial import Voronoi, voronoi_plot_2d
vor = Voronoi(points)

import matplotlib.pyplot as plt
fig = voronoi_plot_2d(vor,point_size=1,show_vertices=False)
plt.xlim([0,1])
plt.ylim([0,1])
plt.show()

