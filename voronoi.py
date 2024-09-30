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


fullData = pd.read_csv("fullData.csv")

nonweird = []
nwnodes = []
nwprobs=[]
for i in range(len(mse_array)):
    if mse_array[i] < 10:
        nonweird.append(mse_array[i])
        nwnodes.append(nodes_array[i])
        nwprobs.append(probs_array[i])

mse_array = np.array(nonweird)
nodes_array = np.array(nwnodes)
probs_array = np.array(nwprobs)

log_array = np.zeros(len(mse_array)) #[] #np.zeros(len(mse_array))
for i in range(len(mse_array)):
    # if np.log(mse_array[i]) > 0:
    #     log_array[i] = 0
    if np.log(mse_array[i]) <= -4:
        log_array[i] = -4 #log_array.append(-4) #log_array[i] = -4
    elif np.log(mse_array[i]) >= 10**.5:
        log_array[i] = 0
        #nodes_array.remove(i)
        #probs_array.(i)

    elif mse_array[i] != 0:
        log_array[i] = np.log(mse_array[i]) #log_array.append(np.log(mse_array[i]))
    
    if mse_array[i] > 10**2:
        print(fullData.iloc[i][0])
        print(fullData.iloc[i][1])
        print("n",nodes_array[i])
        print("p",probs_array[i])
    # else:
    #     #log_array.append(-4)
    #     log_array[i] = -4

mse_array = log_array


points = np.column_stack((nodes_array/100, probs_array))
ring = np.array([[np.cos(x), np.sin(x)] for x in np.linspace(0,7)])*10
ring_z = np.array([1 for x in np.linspace(0,7)])
points = np.concatenate([points, ring])
mse_array = np.concatenate([mse_array, ring_z])

# Generate Voronoi diagram
vor = Voronoi(points)

# Normalize z values to get a colormap
norm = plt.Normalize(vmin=min(mse_array), vmax=-.3)
cmap = cm.gnuplot2_r #cm.viridis

# Create a plot
fig, ax = plt.subplots()

# Plot Voronoi diagram with cells colored based on z values
ax1 = voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='grey', line_width=1, line_alpha=0.6, point_size=0)

# Color each Voronoi region
def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def polygon_area(vertices):
    n = len(vertices)
    area = 0
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]  # Wrap around to the first vertex
        area += x1 * y2 - x2 * y1
    return abs(area) / 2

areas = []
counter = 0
for region_index, region in enumerate(vor.regions):
    if not -1 in region and len(region) > 0:
        print(counter)
        polygon = [vor.vertices[i] for i in region]
        poly_array = np.array(polygon)
        x = poly_array[:,0]
        y = poly_array[:,1]
        color = cmap(norm(mse_array[region_index]))
        ax.fill(*zip(*polygon), color=color)
        area = polygon_area(polygon)
        areas.append(area)
        if area >= .006:
            pt = np.where(vor.point_region == region_index)[0][0]
            plt.plot([vor.points[pt,0]], [vor.points[pt,1]], color = "grey", marker = ".", markersize = 5)
        counter += 1

areas = np.array(areas)
print("mean", np.mean(areas))
print("med", np.median(areas))
print("max", np.max(areas))
print("min", np.min(areas))
        


# Plot the original points
#ax.plot(points[:, 0], points[:, 1], 'ko')



# Add a color bar to show the mapping of z values to colors
sm = plt.cm.ScalarMappable(cmap="gnuplot2_r", norm=norm)
sm.set_array([])

newticks1 = plt.xticks()[0]
#newticks = np.zeros(len(labels))

newticks2 = []
for i in range(len(newticks1)):
    if newticks1[i]==0 or newticks1[i]==-1 or newticks1[i]==-2 or newticks1[i]==-3 or newticks1[i]==-4:
        newticks2.append(r'$10^{{{}}}$'.format(newticks1[i]))
    # else:
    #     newticks2[i] = 
c_bar = plt.colorbar(sm, ax=ax, label=r'$MSE$')
print(newticks2)
# c_bar.ax.set_yticklabels(newticks2)
# for i in range(len(newticks1)):
#     if isinstance(newticks1[i], float):
#         c_bar.ax.set_yticks[i].label1.set_visible(False)
ticks = [-4,-3,-2,-1]
c_bar.set_ticks(ticks)
newticks2 = [r'$10^{{{}}}$'.format(x) for x in ticks]
c_bar.set_ticklabels(newticks2)

def add_colorbar_neg(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    newticks1 = cbar.ax.get_yticklabels()
    newticks1 = [label.get_text() for label in newticks1]
    newticks1 = [a.replace('−', '-') for a in newticks1]
    #newticks1 = [int(a) for a in newticks1 if "." not in a]
    #newticks1 = [float(a) for a in newticks1 if type(a) != int]
    newticks2 = np.zeros(len(newticks1))
    for i in range(newticks1):
        if isinstance(newticks1[i], int):
            newticks2[i] = r'$10^{{{}}}$'.format(newticks1[i])
        else:
            newticks2[i] = ''
    #newticks2 = [r'$10^{{{}}}$'.format(x) for x in newticks1]
    cbar.ax.set_yticklabels(newticks2) 
    plt.sca(last_axes)
    return cbar

#add_colorbar_neg(ax1)

plt.xlabel(r'$N$')
plt.ylabel(r'$p$')
plt.title('Voronoi Diagram Colored by MSE')
plt.xlim([0,1])
plt.ylim([0,1])

# labels= plt.xticks()[0]
# newticks2 = [100*x for x in labels]
# plt.xticks(newticks2)

import matplotlib.ticker as ticker
scale_x = 100
plt.xticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1])
ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x*scale_x))
ax.xaxis.set_major_formatter(ticks_x)

plt.savefig("voronoi.pdf")
plt.savefig("Figure 4")