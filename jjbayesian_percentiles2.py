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


data = pd.read_pickle("bayesian array")
nodes = pd.read_pickle("bayesian nodes")
#print(len(data))

l=len(data[0])


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


nwks_list2 = pd.read_csv("nwks_list2")
fullData = pd.read_csv('fullData.csv')
#print(string2array(fullData.iloc[3][3], sep=" "))

nwks_list2 = pd.read_csv("nwks_list2")
cwd = os.getcwd()
nwks_names = []

data_names = []

for j in range(len(fullData)):
    data_name = fullData.iloc[j][0]
    data_names.append(data_name)


paths = np.zeros(len(data_names),dtype=object)

for i in range(len(nwks_list2)):
    pathname = nwks_list2.iloc[i][1]
    nwks_listname = os.path.basename(nwks_list2.iloc[i][1])
    if nodecount_edge(file_name = pathname) <= 100 and nodecount_edge(file_name = pathname) >= 2:
        nwks_names.append(nwks_listname)
        if nwks_listname in data_names:
            indx = data_names.index(nwks_listname)
            #print(indx)
            paths[indx] = pathname


for i in range(len(paths)):
    if i == 1550:
        paths[i] = r"C:\Users\jj\Downloads\GitHub\small-perc\pholme_networks\tests\krackhardt_kite.adj"
    elif i == 1560:
        paths[i] = r"C:\\Users\jj\Downloads\GitHub\small-perc\pholme_networks\tests\sedgewick_maze.adj"
    elif i == 1564:
        paths[i] = r"C:\\Users\jj\Downloads\GitHub\small-perc\pholme_networks\wss\146.adj"
    elif i ==  1565:
        paths[i] = r"C:\\Users\jj\Downloads\GitHub\small-perc\pholme_networks\wss\147_acct_recip.adj"
    elif i == 1566:
        paths[i] = r"C:\\Users\jj\Downloads\GitHub\small-perc\pholme_networks\wss\147_reject_any.adj"
    elif i == 1567:
        paths[i] = r"C:\\Users\jj\Downloads\GitHub\small-perc\pholme_networks\wss\148.adj"
    elif i == 1568:
        paths[i] = r"C:\\Users\jj\Downloads\GitHub\small-perc\pholme_networks\wss\151.adj"
    elif i == 1569:
        paths[i] = r"C:\\Users\jj\Downloads\GitHub\small-perc\pholme_networks\wss\152.adj"
    elif i == 1570:
        paths[i] = r"C:\\Users\jj\Downloads\GitHub\small-perc\pholme_networks\wss\153.adj"
    elif i == 1571:
        paths[i] = r"C:\\Users\jj\Downloads\GitHub\small-perc\pholme_networks\wss\154.adj"
    elif i == 1572:
        paths[i] = r"C:\\Users\jj\Downloads\GitHub\small-perc\pholme_networks\wss\155.adj"
    elif i == 1573:
        paths[i] = r"C:\\Users\jj\Downloads\GitHub\small-perc\pholme_networks\wss\157.adj"
    elif i == 1574:
        paths[i] = r"C:\\Users\jj\Downloads\GitHub\small-perc\pholme_networks\wss\158.adj"
    elif i == 1575:
        paths[i] = r"C:\\Users\jj\Downloads\GitHub\small-perc\pholme_networks\wss\159.adj"
    elif i == 1576:
        paths[i] = r"C:\\Users\jj\Downloads\GitHub\small-perc\pholme_networks\wss\160.adj"
    elif i == 1578:
        paths[i] = r"C:\\Users\jj\Downloads\GitHub\small-perc\pholme_networks\wss\162.adj"
    elif i == 1579:
        paths[i] = r"C:\\Users\jj\Downloads\GitHub\small-perc\pholme_networks\wss\163.adj"
    elif i == 1580:
        paths[i] = r"C:\\Users\jj\Downloads\GitHub\small-perc\pholme_networks\wss\171.adj"
    elif i == 1581:
        paths[i] = r"C:\\Users\jj\Downloads\GitHub\small-perc\pholme_networks\wss\174.adj"
    elif i == 1582:
        paths[i] = r"C:\\Users\jj\Downloads\GitHub\small-perc\pholme_networks\wss\265.adj"
    elif i == 1583:
        paths[i] = r"C:\\Users\jj\Downloads\GitHub\small-perc\pholme_networks\wss\266.adj"
    elif i == 1584:
        paths[i] = r"C:\\Users\jj\Downloads\GitHub\small-perc\pholme_networks\wss\267.adj"
    elif i == 1585:
        paths[i] = r"C:\\Users\jj\Downloads\GitHub\small-perc\pholme_networks\wss\268.adj"
    elif i == 1586:
        paths[i] = r"C:\\Users\jj\Downloads\GitHub\small-perc\pholme_networks\wss\269.adj"
    elif i == 1587:
        paths[i] = r"C:\\Users\jj\Downloads\GitHub\small-perc\pholme_networks\wss\270.adj"
    elif i == 1588:
        paths[i] = r"C:\\Users\jj\Downloads\GitHub\small-perc\pholme_networks\wss\271.adj"
    elif i == 1589:
        paths[i] = r"C:\\Users\jj\Downloads\GitHub\small-perc\pholme_networks\wss\272.adj"
    elif i == 1590:
        paths[i] = r"C:\\Users\jj\Downloads\GitHub\small-perc\pholme_networks\wss\273.adj"
    elif i == 1591:
        paths[i] = r"C:\\Users\jj\Downloads\GitHub\small-perc\pholme_networks\wss\275.adj"
    elif i == 1592:
        paths[i] = r"C:\\Users\jj\Downloads\GitHub\small-perc\pholme_networks\wss\276.adj"
    elif i == 1593:
        paths[i] = r"C:\\Users\jj\Downloads\GitHub\small-perc\pholme_networks\wss\277.adj"
    elif i == 1594:
        paths[i] = r"C:\\Users\jj\Downloads\GitHub\small-perc\pholme_networks\wss\278.adj"
    elif i == 1595:
        paths[i] = r"C:\\Users\jj\Downloads\GitHub\small-perc\pholme_networks\wss\282.adj"
    elif i == 1596:
        paths[i] = r"C:\\Users\jj\Downloads\GitHub\small-perc\pholme_networks\wss\304.adj"
    elif i == 1597:
        paths[i] = r"C:\\Users\jj\Downloads\GitHub\small-perc\pholme_networks\wss\395.adj"
    elif i == 1598:
        paths[i] = r"C:\\Users\jj\Downloads\GitHub\small-perc\pholme_networks\wss\396.adj"
    elif i == 1600:
        paths[i] = r"C:\\Users\jj\Downloads\GitHub\small-perc\pholme_networks\wss\428.adj"
    elif i == 1601:
        paths[i] = r"C:\\Users\jj\Downloads\GitHub\small-perc\pholme_networks\wss\518.adj"
    elif i == 1602:
        paths[i] = r"C:\\Users\jj\Downloads\GitHub\small-perc\pholme_networks\wss\519.adj"
    elif i == 1603:
        paths[i] = r"C:\\Users\jj\Downloads\GitHub\small-perc\pholme_networks\wss\521.adj"
    elif i == 1604:
        paths[i] = r"C:\Users\jj\Downloads\GitHub\small-perc\pholme_networks\wss\522.adj"
    elif i == 1605:
        paths[i] = r"C:\Users\jj\Downloads\GitHub\small-perc\pholme_networks\wss\524.adj"
    elif i == 1606:
        paths[i] = r"C:\Users\jj\Downloads\GitHub\small-perc\pholme_networks\wss\525.adj"
    elif i == 1607:
        paths[i] = r"C:\Users\jj\Downloads\GitHub\small-perc\pholme_networks\wss\636.adj"
    elif i == 1608:
        paths[i] = r"C:\Users\jj\Downloads\GitHub\small-perc\pholme_networks\wss\637.adj"


print("paths")
print(paths)
for i in range(len(paths)):
    if paths[i] == 0:
        print(i)
# print(nwks_list2.iloc[0])
cwd = os.getcwd() 
# print("Current working directory:", cwd) 

def r_pow(x, n, d):
    """
    Compute x to the power of n/d (not reduced to lowest
    expression) with the correct function real domains.
    
    ARGS:
        x (int,float,array): base
        n (int)            : exponent numerator
        d (int)            : exponent denominator
        
    RETURNS:
        x to the power of n/d
    """
    
    # list to array
    if type(x) == list:
        x = np.array(x)
    # check inputs
    if type(n) != int or type(d) != int:
        raise Exception("Exponent numerator and denominator must be integers")
    # if denominator is zero
    if not d:
        raise Exception("Exponent denominator cannot be 0")
        
    # raise x to power of n
    X = x**n
    # even denominator
    if not d % 2:
        # domain is for X>=0 only
        if type(x) == np.ndarray:
            X[X<0] = np.nan
        elif X < 0:
            X = np.nan
        res = np.power(X, 1./d)
        return res
    # odd denominator
    else:
        # domain is all R
        res = np.power(np.abs(X), 1./d)
        res *= np.sign(X)
        return res
    

roots = [r_pow(data[0][i], 1, int(nodes[0][i])) for i in range(l)] 

#plt.plot(nodes[0], roots, lw=0, marker='x')
top20 = np.array([[nodes[0][i], roots[i]] for i in range(l) if roots[i]>=np.percentile(roots,80)]).T
plt.plot(top20[0], top20[1], lw=0, marker='x')
plt.show()
bottom20 = np.array([[nodes[0][i], roots[i]] for i in range(l) if roots[i]<np.percentile(roots,20)]).T
plt.plot(bottom20[0], bottom20[1], lw=0, marker='x')
plt.show()

bottom20_i = np.array([i for i in range(l) if roots[i]<np.percentile(roots,20)]).T
top20_i = np.array([i for i in range(l) if roots[i]>=np.percentile(roots,80)]).T

#print(bottom20_i)
#print(top20_i)




#fullData = pd.read_csv("fullData.csv")
#counter_100 = 0

y_array_b = []
y_array_t = []
bottom_nodes = []
top_nodes = []

import os

indices = pd.read_pickle("bayesian indices")
#print("indices")
#print(indices)
#print(indices[0])
#print('iloc')
#print(indices.iloc[0][0])
 

# nwks_100 = []
# for i in range(len(nwks_list2)):
#     if nwks_list2.iloc[i][1] == 100:
#         nwks_100.append(i)

# print(nwks_100)

#counter_100 = 0
for i_b in range(len(bottom20_i)):
    #print("i_b",i_b)
    #counter_100 = 0
    # for j in nwks_100:
    #     if j < i_b:
    #         counter_100 += 1
    #new_ib = indices.iloc[bottom20_i[i_b]-counter_100][0]
    new_ib = bottom20_i[i_b]
    #print("file1",nwks_list2.iloc[i_b][1])
    file = paths[new_ib]
    print(file)
    #if nodecount_edge(file_name = file) == 100:
        #counter_100 += 1
        #print("counter", counter_100)
    #if nodecount_edge(file_name = file) <= 100:
        #file2 = file.replace("C:\\Users\\jj\Downloads\\GitHub\small-perc\\pholme_networks", '')
        #print("file2",file)
        #file = file[60:]
        #print("file3",file2)
        # bottom_nodes.append(nodecount_edge(file_name=file))
        #values = mega_file_reader2(removal = "random", adj_list = [file])
    #   print(file)
    #   print("val",values)
    #   print("averaged data",values[0])
    #   print("fin", values[1])
        #print("nodes")
        #print(fullData.iloc[bottom20_i[i_b]-counter_100][1])
        #print(nodecount_edge(file_name=file))

    if fullData.iloc[new_ib][1] != nodecount_edge(file_name=file):
            print("nodes")
            print(fullData.iloc[new_ib][1])
            print(nodecount_edge(file_name=file))
    sim = string2array(fullData.iloc[new_ib][3], sep=" ")
    fin = string2array(fullData.iloc[new_ib][5], sep=" ")
    y = ((fin - sim) ** 2).mean()
    if y < 100:
        y_array_b.append(y)
        bottom_nodes.append(nodecount_edge(file_name=file))

y_array_bdf = pd.DataFrame(y_array_b)
y_array_bdf.to_pickle("bayesian mse bottom 20 (n=100)")

#counter_100 = 0
for i_t in range(len(top20_i)):
    #print("i_t",i_t)
    #counter_100 = 0
    #print("counter", counter_100)
    # for j in nwks_100:
    #     if j < i_b:
    #         counter_100 += 1
    #new_it = indices.iloc[top20_i[i_t]-counter_100][0]
    new_it = top20_i[i_t]
    file = paths[new_it]

    #if nodecount_edge(file_name = file) < 100:
    sim = string2array(fullData.iloc[top20_i[i_t]][3], sep=" ")
    fin = string2array(fullData.iloc[top20_i[i_t]][5], sep=" ")

    if fullData.iloc[top20_i[i_t]][1] != nodecount_edge(file_name=file):
          print("nodes")
          print(fullData.iloc[top20_i[i_t]][1])
          print(nodecount_edge(file_name=file))
    y = ((fin - sim) ** 2).mean()
    if y < 100:
        y_array_t.append(y)
        top_nodes.append(nodecount_edge(file_name=file))

y_array_tdf = pd.DataFrame(y_array_t)
y_array_tdf.to_pickle("bayesian mse top 20 (n=100)")

bottom_nodesdf = pd.DataFrame(bottom_nodes)
bottom_nodesdf.to_pickle("bottom nodes (n=100)")

top_nodesdf = pd.DataFrame(top_nodes)
top_nodesdf.to_pickle("top nodes (n=100)")

#print("total indices",len(bottom_nodes)+len(top_nodes))

# top_indices = pd.read_pickle("bayesian mse top 20")
# bottom_indices = pd.read_pickle("bayesian mse bottom 20")

# bottom = pd.read_pickle("bottom nodes")
# top = pd.read_pickle('top nodes')

# xb = np.zeros(len(bottom))
# xt = np.zeros(len(top))

# print("lenbottom",len(bottom))
# for k in range(len(bottom)):  
#   print(bottom.iloc[k][0])
#   xb[k] = bottom.iloc[k][0]

# for l in range(len(top)):
#     xt[l] = top.iloc[l][0]



# top_y = np.zeros(len(top_indices))
# bot_y = np.zeros(len(bottom_indices))


# print('bot')
# print(bottom_indices.iloc[0][0])

# for i in range(len(top_indices)):
#   top_y[i] = top_indices.iloc[i][0]


# for j in range(len(bottom_indices)):
#   bot_y[j] = bottom_indices.iloc[j][0]

# print("boty")
# print(bot_y)




plt.plot(top_nodes,y_array_t,'x',color= "red", label="top 20")
plt.plot(bottom_nodes,y_array_b,'x',color= "blue", label="bottom 20")
plt.xlabel("nodes")
plt.ylabel('mse')
plt.legend()
plt.show()

mean_top = np.mean(y_array_t)
print(mean_top)
mean_bottom = np.mean(y_array_b)
print(mean_bottom)



# fullData = pd.read_csv("fullData.csv")
# nwks_list2 = pd.read_csv("nwks_list2")
# cwd = os.getcwd()


# nwks_names = []
# data_names = []


# for j in range(len(fullData)):
#     data_name = fullData.iloc[j][0]
#     data_names.append(data_name)

# missing = []

# for i in range(len(nwks_list2)):
#     pathname = nwks_list2.iloc[i][1]
#     nwks_listname = os.path.basename(nwks_list2.iloc[i][1])
#     if nodecount_edge(file_name = pathname) <= 100 and nodecount_edge(file_name = pathname) >= 2:
#         nwks_names.append(nwks_listname)
#         if nwks_listname not in data_names:
#             if nwks_listname not in missing:
#                 #print(pathname)
#                 #print(nwks_listname)
#                 missing.append(pathname)
# # print(len(data_names))
# # print(len(nwks_names))
# # print(len(missing))


# def read_from_adj(filename):
    
#     file = open(filename, "r")
#     content = file.readlines()

#     # convert into networkx graph
#     node_list = []
#     edge_list = [] #np.empty(len(content), dtype=object)
    
#     if len(content) == 0:
#         G = nx.Graph()
#         return G
    
#     edge_count = 0
#     for i in range(len(content)):
        
#         edge = content[i].strip()
#         edge = edge.split(" ")
        
#         if len(edge)==2:
            
#             edge_list.append([int(edge[0]), int(edge[1])])
#             node_list.append(int(edge[0]))
#             node_list.append(int(edge[1]))

#     node_list = list(set(node_list))
    
#     if 0 in node_list:
#         n = max(node_list) + 1
#         offset = 0
#     else:
#         n = max(node_list)
#         offset = min(node_list)
        
#     adj = np.zeros((n, n))
        
#     for k in range(len(edge_list)):
#         adj[int(edge_list[k][0])-offset, int(edge_list[k][1])-offset] = 1
#         adj[int(edge_list[k][1])-offset, int(edge_list[k][0])-offset] = 1

#     G = nx.from_numpy_array(adj)
#     file.close()
            
#     return G

    
# def random_removal(G0):
    
#     # make a copy of input graph
#     G = G0.copy()
#     n = G.number_of_nodes()
    
#     data_array = np.zeros(n, dtype=float)
    
#     for i in range(n):
#         # get LCC size
#         data_array[i] = len(max(nx.connected_components(G), key=len)) / (n - i)
#         # find a random node to remove
#         if G.number_of_nodes() != 0:
#             v = choice(list(G.nodes()))
#             G.remove_node(v)
            
#     return data_array

            
# def targeted_removal(G0):
    
#     # make a copy of input graph
#     G = G0.copy()
#     n = G.number_of_nodes()
    
#     data_array = np.zeros(n, dtype=float)
#     for i in range(n):
#         # get LCC size
#         data_array[i] = len(max(nx.connected_components(G), key=len)) / (n - i)
#         # find highest-degree node and remove it
#         if G.number_of_nodes() != 0:
#             v = sorted(G.degree, key=lambda x: x[1], reverse=True)[0][0]
#             G.remove_node(v)
            
#     return data_array


# def fullDataTable(nwks_list, num_tries=100, max_size=100, min_counter=0):

#     table = np.zeros((len(nwks_list),7), dtype=object)
    
#     counter = 0
#     for i, nwpath in enumerate(nwks_list):
        
#         # extract file name from file path
#         nwname = os.path.basename(nwpath)
#         # add name of network to table
#         table[counter,0] =  str(nwname)
#         # read graph from ".adj" file
#         print('{} {}'.format(i, nwname), end='')
#         G = read_from_adj(nwpath)
#         # set p for G(n,p) graph
#         n = G.number_of_nodes()
#         m = G.number_of_edges()
#         p = m / scipy.special.comb(n, 2)
#         print(' has (n,m) = ({}, {})'.format(n, m), end='')
        
#         # check if network meets size limitation
#         if n > max_size:
#             print (' --- omit')
#             continue
#         elif n < 2:
#             print(' --- omit')
#             continue
#         else:
#             print(' --- compute', end='')

#         if counter >= min_counter:
#             t0 = time.time()
#             # add number of nodes and edges to info table
#             table[counter,1] = n
#             table[counter,2] = m
    
#             # get data for random and targeted node removal 
#             nw_r = np.nanmean([random_removal(G) for i in range(num_tries)], axis=0)
#             nw_t = targeted_removal(G)
            
#             # finite-theory results for random and targeted node removal
#             theory_r = relSCurve(p, n, attack=False, reverse=True, lcc_method_relS="pmult", executable_path='libs/p-recursion-float128.exe')
#             theory_t = relSCurve(p, n, attack=True, reverse=True, lcc_method_relS="pmult", executable_path='libs/p-recursion-float128.exe')
        
#             # rel LCC arrays
#             results = [nw_r, nw_t, theory_r, theory_t]
#             for i, array in enumerate(results): 
#                 # store in info table
#                 table[counter,3+i] = array
    
    
#             # with open('data/fulldata-{}.txt'.format(counter), 'w') as file:
#             #     # Write four lines to the file
#             #     file.write("{} {} {}\n".format(nwname, n, m))
#             #     file.write(' '.join(map(str, nw_r))+"\n")
#             #     file.write(' '.join(map(str, nw_t))+"\n")
#             #     file.write(' '.join(map(str, theory_r))+"\n")
#             #     file.write(' '.join(map(str, theory_t))+"\n")

#             # print(' in {} s'.format(time.time()-t0))
        
#         counter+=1

#     if min_counter==0:
#         # remove empty rows from table
#         table2 = table[:counter]
    
#         # convert to data frame and name its columns
#         df = pd.DataFrame(table2)
#         df.columns = ["network", "nodes", "edges", "real rand rLCC", "real attack rLCC",
#                       "fin theory rand rLCC", "fin theory attack rLCC"]
#         #print(info_table)
#         return df
#     else:
#         return 


# df = fullDataTable(missing,100,100,0)
# df.to_csv('fullData.csv', mode='a', index=False, header=False)




# new_data = np.array((len(missing),7),dtype=object)
# for i in range(len(missing)):
#     pathname = missing[i]
#     new_data[i][0] = os.path.basename(pathname)
#     new_data[i][1] = nodecount_edge(pathname)

#     file = open(pathname, "r")
#     #content = file.readlines()
#     content = (line.rstrip() for line in file)  # All lines including the blank ones
#     content = list(line for line in content if line)
#     new_data[i][2] = len(content)

#     p = 


# for j in data_names:
#     if j not in nwks_names:
#         missing.append(j)
# print(len(missing))

#print(missing)


# datacounter = 0

# for i in range(len(nwks_list2)):
#     nwks_listname = os.path.basename(nwks_list2.iloc[i][1])
#     #print("nwk",nwks_listname)
#     fullDataname = fullData.iloc[datacounter][0]
#     #print("data",fullDataname)
#     if nwks_listname != fullDataname:
#         print("i",i)
#         print("nwk",nwks_listname)
#         datacounter -= 1
#     datacounter += 1


