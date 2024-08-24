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

nonweird = []
for data in mse_array:
    if data < 10:
        nonweird.append(data)

log_array = np.zeros(len(mse_array))
for i in range(len(mse_array)):
    if mse_array[i] != 0:
        log_array[i] = np.log(mse_array[i])
    else:
        log_array[i] = -100
## histogram
plt.hist(log_array, density = True, log=True, bins=200)
plt.xlim([-30,0])
plt.show()


## voronoi
from scipy.special import comb, lambertw
import numpy as np


#points = []

# for i in range(len(nodes_array)):
#     point = []
#     point.append(nodes_array[i])
#     point.append(probs_array[i])
#     point.append(mse_array[i])
#     points.append(point)


import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.cm as cm

# Sample data: positions (x, y) and corresponding z values
points = np.random.random(size=(600,2)) #np.array([[1, 1], [2, 3], [3, 1], [4, 4], [5, 2]])
z_values = np.random.random(size=600)  # Corresponding z values

# Generate Voronoi diagram
vor = Voronoi(points)

# Normalize z values to get a colormap
norm = plt.Normalize(vmin=min(z_values), vmax=max(z_values))
cmap = cm.viridis

# Create a plot
fig, ax = plt.subplots()

# Plot Voronoi diagram with cells colored based on z values
voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='black', line_width=2, line_alpha=0.6, point_size=1)

# Color each Voronoi region
for region_index, region in enumerate(vor.regions):
    if not -1 in region and len(region) > 0:
        polygon = [vor.vertices[i] for i in region]
        color = cmap(norm(z_values[region_index]))
        ax.fill(*zip(*polygon), color=color)

# Plot the original points
ax.plot(points[:, 0], points[:, 1], 'ko')

# Add a color bar to show the mapping of z values to colors
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, ax=ax, label='Z value')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Voronoi Diagram Colored by Z Values')
#plt.show()

# points = np.column_stack((nodes_array.flatten(),probs_array.flatten(),mse_array.flatten()))

# from scipy.spatial import Voronoi, voronoi_plot_2d

# # import KDTree for the nearest neighbor interpolation
# from scipy.spatial import KDTree, ConvexHull

# # create a densely spaced grid in the plot region

# # do the nearest neighbor interpolation
# kdt = KDTree(points)
# nndist,nnidx = kdt.query(points)
# # nnidx gives the index of the Voronoi nucelus for each point in the grid

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # now cycle through all Voronoi indices
# for voronoi_idx in np.unique(nnidx):
#     idx_polygon = nnidx==voronoi_idx
#     # create a convex hull around the point cloud
#     hull = ConvexHull(points[idx_polygon])    #[idx_polygon]
#     polygon = plt.Poly3DCollection(hull.points[hull.simplices], alpha=0.5, 
#                                facecolors=np.random.uniform(0,1,3),
#                                linewidths=0.5,edgecolors='gray')
#     ax.add_collection3d(polygon)
    
# plt.show()


# voronoi = Voronoi(points)
# voronoi_vertices = voronoi.vertices
# regions = voronoi.regions

# from generativepy.drawing import make_image, setup
# from generativepy.geometry import Circle, Polygon
# from generativepy.color import Color
# from scipy.spatial import Voronoi
# import random

# SIZE = 400
# POINTS = 20

# # Create a list of random points
# random.seed(40)
# points = [[random.randrange(SIZE), random.randrange(SIZE)]
#           for i in range(POINTS)]
# points.append((-SIZE*3, -SIZE*3))
# points.append((-SIZE*3, SIZE*4))
# points.append((SIZE*4, -SIZE*3))
# points.append((SIZE*4, SIZE*4))


# def draw(ctx, pixel_width, pixel_height, frame_no, frame_count):
#     setup(ctx, pixel_width, pixel_height, background=Color(1))
#     voronoi = Voronoi(points)
#     voronoi_vertices = voronoi.vertices

#     for region in voronoi.regions:
#        if -1 not in region:
#            polygon = [voronoi_vertices[p] for p in region]
#            Polygon(ctx).of_points(polygon).stroke(line_width=2)


# make_image("voronoi-lines.png", draw, SIZE, SIZE)

# import matplotlib.pyplot as plt
# fig = voronoi_plot_2d(voronoi)
# plt.show()



## 2D stuff
    # n vs p, colors mse
# max = np.max(mse_array)
# print(max)
# med = np.median(mse_array)
# print(med)

#     # n vs p with mse as colors
# fig , (ax1) = plt.subplots(1, 1)
# ax1plot = ax1.scatter(nodes_array, probs_array , c=mse_array, s=4, marker="x", linewidth = 1, vmin = -.05, vmax = .1, cmap = "Reds")
# #add_colorbar(ax1plot)
# plt.show()

    # mse vs p, bin by nodes
#ax = plt.axes(projection = "3d")
#plt.scatter(probs_array, mse_array, nodes_array)
# plt.scatter(nodes_array, probs_array, mse_array)
# nbin_means,nbin_edges, nbinnumber = scipy.stats.binned_statistic(nodes_array, mse_array, statistic='mean', bins=10, range=None)
# print(nbin_means)
# print(nbin_edges)
# plt.hlines(nbin_means, nbin_edges[:-1], nbin_edges[1:], colors='g', lw=2,
#            label='bins')
# plt.show()

## 3D stuff
#ax = plt.axes(projection ='3d')
 
#ax.scatter(nodes_array, probs_array, mse_array)

#tri = mtri.Triangulation(nodes_array, probs_array)
#ax.plot_trisurf(nodes_array, probs_array, mse_array, triangles=tri.triangles, cmap=plt.cm.Spectral)

#ax.set_title('MSE over n and p')
#plt.show()

