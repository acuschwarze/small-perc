# import sys, pickle
# sys.path.insert(0, "libs")

# import os, pickle, csv # import packages for file I/O
# import time # package to help keep track of calculation time

# import numpy as np
# import networkx as nx
# import matplotlib.pyplot as plt
# import pandas as pd

# import scipy
# import scipy.stats as sst
# from scipy.special import comb
# from scipy.integrate import simpson
# from scipy.signal import argrelextrema
# from random import choice
# from matplotlib.gridspec import GridSpec

# from libs.utils import *
# from libs.finiteTheory import *
# from visualizations import *
# from libs.utils import *
# from robustnessSimulations import *
# from performanceMeasures import *
# from infiniteTheory import *
# from finiteTheory import *

# fvals = pickle.load(open('data/fvalues.p', 'rb'))
# pvals = pickle.load(open('data/Pvalues.p', 'rb'))



# def add_colorbar_neg(mappable):
#     from mpl_toolkits.axes_grid1 import make_axes_locatable
#     import matplotlib.pyplot as plt
#     last_axes = plt.gca()
#     ax = mappable.axes
#     fig = ax.figure
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="5%", pad=0.05)
#     cbar = fig.colorbar(mappable, cax=cax)
#     newticks1 = cbar.ax.get_yticklabels()
#     newticks1 = [label.get_text() for label in newticks1]
#     newticks1 = [a.replace('−', '-') for a in newticks1]
#     #newticks1 = [int(a) for a in newticks1 if "." not in a]
#     #newticks1 = [float(a) for a in newticks1 if type(a) != int]
#     newticks2 = [r'$10^{{{}}}$'.format(x) for x in newticks1]
#     cbar.ax.set_yticklabels(newticks2) 
#     plt.sca(last_axes)
#     return cbar

# def add_colorbar_norm(mappable):
#     from mpl_toolkits.axes_grid1 import make_axes_locatable
#     import matplotlib.pyplot as plt
#     last_axes = plt.gca()
#     ax = mappable.axes
#     fig = ax.figure
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="5%", pad=0.05)
#     cbar = fig.colorbar(mappable, cax=cax) 
#     plt.sca(last_axes)
#     return cbar

# max_n = [10,15,20,25,50,100]
# total_p = 100
# #nodes_array = np.arange(2,max_n+1)
# #probs_array = np.linspace(.01,1,total_p)

# n_threshold = .2
# nodes_list = [10,15,25,50,100]
# probs_array = [(1/(n_threshold*(x-1))) for x in max_n]

# def fig_3(max_n,probs_array):
#     nodes_array = max_n
#     # probs_array = np.linspace(.01,1,total_p)

#     for ix in range(len(nodes_array)):
#         i = nodes_array[ix]
#         probs = probs_array[ix]
#         r_all_sim = relSCurve_precalculated(i, probs, targeted_removal=False, simulated=True, finite=False)
#         r_sim = np.zeros(i)
#         for k in range(i):
#             r_sim = r_sim[:50] + np.transpose(r_all_sim[:,k][:i][:50])
#         r_sim = r_sim / i

#         r_fin = (relSCurve_precalculated(i, probs, targeted_removal=False, simulated=False, finite=True)[:i])[:50]
        
#         r_inf = infiniteTheory.relSCurve(i, probs, attack=False, smooth_end=False)[:50]


#         t_all_sim = relSCurve_precalculated(i, probs, targeted_removal=True, simulated=True, finite=False)
#         t_sim = np.zeros(i)
#         for k in range(i):
#             t_sim = t_sim + np.transpose(t_all_sim[:,k][:i])
#         t_sim = t_sim / i

#         t_fin = (relSCurve_precalculated(i, probs, targeted_removal=True, simulated=False, finite=True)[:i])
        
#         t_inf = infiniteTheory.relSCurve(i, probs, attack=True, smooth_end=False)

#         # heatmap_rfin[j][i-2] = ((r_fin-r_sim)**2).mean() * i
#         # heatmap_rinf[j][i-2] = ((r_inf-r_sim)**2).mean() * i
#         # heatmap_tfin[j][i-2] = ((t_fin-t_sim)**2).mean() * i
#         # heatmap_tinf[j][i-2] = ((t_inf-t_sim)**2).mean() * i

#         print(i,"fin rand", "mse",((r_fin-r_sim)**2).mean())

#         print(i,"inf rand", "mse",((r_inf-r_sim)**2).mean())

#         print(i, "fin tar", "mse",((t_fin-t_sim)**2).mean())

#         print(i,"inf tar","mse",((t_inf-t_sim)**2).mean())

    
# fig_3(max_n,probs_array)


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

cwd = os.getcwd() 
print("Current working directory:", cwd) 


from fnmatch import fnmatch

root = r'C:\Users\jj\Downloads\GitHub\small-perc\pholme_networks'
pattern = "*.adj"
pattern2 = "*.arc"
nwks_list2 = []

for path, subdirs, files in os.walk(root):
    for name in files:
        if fnmatch(name, pattern):
            # print(os.path.join(path, name))
            nwks_list2.append(os.path.join(path, name))
        elif fnmatch(name, pattern2):
            nwks_list2.append(os.path.join(path, name))



def nodecount_edge(file_name = ""):
    file = open(file_name, "r")
    #content = file.readlines()
    content = (line.rstrip() for line in file)  # All lines including the blank ones
    content = list(line for line in content if line)
    if len(content)==0:
        return 0
    #print(content)
    node_list = []
    edge_list = np.empty(len(content), dtype=object)
    for i in range(len(content)):
        edge = content[i].strip()
        edge = edge.split(" ")
        edge_list[i] = np.zeros(2)
        #print("i", i)
        #print("edge[0]",edge[0])
        edge_list[i][0] = int(edge[0])
        edge_list[i][1] = int(edge[1])
        for j in range(2):
            node_list.append(int(edge[j]))
    if 0 in node_list:
        n = max(node_list) + 1
    else:
        n = max(node_list)
    return n


fvals = pickle.load(open('data/fvalues.p', 'rb'))
pvals = pickle.load(open('data/Pvalues.p', 'rb'))


def get_full_path(relative_path):
    """Gets the full path from a relative path."""
    return os.path.abspath(relative_path)
    #return os.path.abspath("pholme_networks\\" + relative_path)

# import os
# os.chdir(r'C:\Users\jj\Downloads\GitHub\small-perc\pholme_networks')

fullData = pd.read_csv("fullData.csv")
bigerror = pd.read_csv("bigvalsreal.csv")

nwks=[]

# for i in range(len(nwks_list2)):
#     nwname = os.path.basename(nwks_list2[i])
#     for j in range(len(fullData)):
#         if nwname == fullData.iloc[j][0]:
#             nwks.append(nwks_list2[i])


#****
# for j in range(len(bigerror)):
#     dataidx = np.where(fullData["sourceFileName"] == bigerror.iloc[j][1])[0][0]
#     #print('dataidx',dataidx)
#     for i in range(len(nwks_list2)):
#         nwname = os.path.basename(nwks_list2[i])
#         #print(nwname)
#         #print(fullData.iloc[dataidx][0])
#         if nwname == fullData.iloc[dataidx][0]:
#             nwks.append(nwks_list2[i])
#             break

# print(len(nwks))

def check_space(string):
    '''Check if there is a space in a string to help identify edge list files.'''
    
    # counter
    count = 0

    # loop for search each index
    for i in range(0, len(string)):

        # Check each char
        # is blank or not
        if string[i] == " ":
            count += 1

    return count

def mega_file_reader(theory = False, removal = "random", adj_list = nwks, oneplot = False, num_trials = 100):
    for file_name in adj_list:
        print(file_name)
        file = open(file_name, "r")
        #content = file.readlines()
        content = (line.rstrip() for line in file)  # All lines including the blank ones
        content = list(line for line in content if line)
        #print("linecount")
        #print(len(content))
        #print(len(content[0]))
        #print(content[0])
        if len(content) == 0:
            file.close()
            print("0")
        elif nodecount_edge(file_name) > 100:
                print("over 100")
                file.close()
        else:
            node_list = []
            edge_list = np.empty(len(content), dtype=object)
            for i in range(len(content)):
                edge = content[i].strip()
                edge = edge.split(" ")
                edge_list[i] = np.zeros(2)
                edge_list[i][0] = int(edge[0])
                edge_list[i][1] = int(edge[1])
                for j in range(2):
                    node_list.append(int(edge[j]))
            if 0 in node_list:
                n = max(node_list) + 1
            else:
                n = max(node_list)
            adj = np.zeros((n, n))

            for k in range(len(edge_list)):
                if 0 in node_list:
                    adj[int(edge_list[k][0]), int(edge_list[k][1])] = 1
                    adj[int(edge_list[k][1]), int(edge_list[k][0])] = 1
                else:
                    adj[int(edge_list[k][0]-1), int(edge_list[k][1]-1)] = 1
                    adj[int(edge_list[k][1]-1), int(edge_list[k][0]-1)] = 1

            G_0 = nx.from_numpy_array(adj)
            # G_0 = nx.algorithms.bipartite.matrix.from_biadjacency_matrix(adj, create_using=None)
            G = G_0.copy()
            #nx.draw(G)
            #plt.show()
            averaged_data = np.zeros(n)
            for j_2 in range(num_trials):
                G = G_0.copy()
                # print(list(G.nodes()), "nodes")
                data_array = np.zeros(n, dtype=float)
                for i_2 in range(n):
                    #print(G.number_of_nodes(), "g size before")
                    data_array[i_2] = len(max(nx.connected_components(G), key=len)) / (n - i_2)
                    # find a node to remove
                    if removal == "random":
                        if G.number_of_nodes() != 0:
                            v = choice(list(G.nodes()))
                            G.remove_node(v)
                            # print(v)
                    elif removal == "attack":
                        if G.number_of_nodes() != 0:
                            v = sorted(G.degree, key=lambda x: x[1], reverse=True)[0][0]
                            G.remove_node(v)
                averaged_data += data_array
            averaged_data /= num_trials
            #print(averaged_data, "y")
    return averaged_data


# k = len(nwks)

# mse_array = np.zeros((k,4),dtype=object)

# for i in range(k):
#     dataidx2 = np.where(fullData["sourceFileName"] == bigerror.iloc[i][1])[0][0]
#     print("dataidx2",dataidx2)
    
#     n = fullData.iloc[dataidx2][1]
#     p = fullData.iloc[dataidx2][2] / scipy.special.comb(n,2)
#     sim = mega_file_reader(theory = False, removal = "attack", adj_list = [nwks[i]], oneplot = False, num_trials = 100)
#     fin = string2array(fullData.iloc[dataidx2][5], sep=" ")
#     plt.plot(sim, label = "simulated G(" + str(n) + "," + str(p) + ")")
#     plt.plot(fin, label = "Srec G(" + str(n) + "," + str(p) + ")")
#     plt.title(fullData.iloc[dataidx2][0] + "||" + os.path.basename(nwks[i]), fontsize = 10)
#     #plt.title(os.path.basename(nwks[i]),fontsize = 10)
#     plt.legend()
#     plt.show()
#     mse = ((fin-sim)**2).mean()


#     mse_array[i][0] = fullData.iloc[i][0]
#     mse_array[i][1] = n
#     mse_array[i][2] = p
#     mse_array[i][3] = mse


#test = finiteTheory.relSCurve(.0486, 62,
                                       # attack=False, fdict=fvals, pdict=pvals, lcc_method_relS="pmult", executable_path='libs/p-recursion-float128.exe')
#plt.plot(test)
#plt.show()


# fig = plot_graphs(numbers_of_nodes=[75], edge_probabilities=[0.0598],
#     graph_types=['ER'], remove_strategies=['random'],
#     performance='relative LCC', num_trials=100,
#     smooth_end=False, forbidden_values=[], fdict=fvals, pdict=pvals, lcc_method_main = "pmult", savefig='', simbool = True, executable_path = 'libs/p-recursion-float128.exe', executable2 = 'libs/p-recursion-float128.exe')
# plt.show()

for j in range(len(fullData)):
    for i in range(len(nwks_list2)):
        nwname = os.path.basename(nwks_list2[i])
        if nwname == fullData.iloc[j][0]:
            nwks.append(nwks_list2[i])
            break

k = len(nwks)

for i in range(k):
    
    n = fullData.iloc[i][1]
    p = fullData.iloc[i][2] / scipy.special.comb(n,2)
    sim = mega_file_reader(theory = False, removal = "attack", adj_list = [nwks[i]], oneplot = False, num_trials = 100)
    fin = string2array(fullData.iloc[i][5], sep=" ")
    plt.plot(sim, label = "simulated G(" + str(n) + "," + str(p) + ")")
    plt.plot(fin, label = "Srec G(" + str(n) + "," + str(p) + ")")
    plt.title(fullData.iloc[i][0] + "||" + os.path.basename(nwks[i]), fontsize = 10)
    #plt.title(os.path.basename(nwks_list2[i]),fontsize = 10)
    plt.legend()
    plt.show()