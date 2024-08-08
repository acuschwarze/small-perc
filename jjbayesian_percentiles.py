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

nwks_list2 = pd.read_csv("nwks_list2")
print(nwks_list2.iloc[0])
cwd = os.getcwd() 
print("Current working directory:", cwd) 




def nodecount_edge(file_name = ""):
    file = open(file_name, "r")
    #content = file.readlines()
    content = (line.rstrip() for line in file)  # All lines including the blank ones
    content = list(line for line in content if line)
    print(content)
    node_list = []
    edge_list = np.empty(len(content), dtype=object)
    for i in range(len(content)):
        edge = content[i].strip()
        edge = edge.split(" ")
        edge_list[i] = np.zeros(2)
        print("i", i)
        print("edge[0]",edge[0])
        edge_list[i][0] = int(edge[0])
        edge_list[i][1] = int(edge[1])
        for j in range(2):
            node_list.append(int(edge[j]))
    if 0 in node_list:
        n = max(node_list) + 1
    else:
        n = max(node_list)
    return n


def nodecount_bi(file_name = ""):
    file = open(file_name, "r")
    content = file.readlines()
    n = len(content)
    if len(content[0]) > n:
        n = len(content[0])
    return n

def check_space(string):
    # counter
    count = 0

    # loop for search each index
    for i in range(0, len(string)):

        # Check each char
        # is blank or not
        if string[i] == " ":
            count += 1

    return count


def mega_file_reader2(removal = "random", adj_list = ["taro.txt"]):
    #, file_path = "C:\\Users\\jj\Downloads\\GitHub\small-perc\\pholme_networks" )
    if removal == "random":
        remove_bool = False
    elif removal == "attack":
        remove_bool = True
        
    for file_name in adj_list:

        file = open(file_name, "r")
        #file = open(os.path.join(file_path,file_name), "r")
        content=file.readlines()

        if len(content) == 0:
            file.close()
            averaged_data = "None"
            fin = "None"
        # identify type of file (biadjacency matrix, edge list)
            # edge list
        elif check_space(content[0])==1 and (len(content) == 1 or len(content) > 2):
            print('edge file')
            if nodecount_edge(file_name) > 100:
                file.close()
                averaged_data = "None"
                fin = "None"
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
                n = max(node_list) + 1
                adj = np.zeros((n, n))

                for k in range(len(edge_list)):
                    adj[int(edge_list[k][0]), int(edge_list[k][1])] = 1
                    adj[int(edge_list[k][1]), int(edge_list[k][0])] = 1
                
                G_0 = nx.from_numpy_array(adj)
                # G_0 = nx.algorithms.bipartite.matrix.from_biadjacency_matrix(adj, create_using=None)
                G = G_0.copy()
                p = len(edge_list) / scipy.special.comb(n, 2)
                averaged_data = np.zeros(n)
                print("average")
                for j_2 in range(100):
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
                #print(x_array, "xarray")
                averaged_data /= 100
                #print(averaged_data, "y")

                fin = finiteTheory.relSCurve(p,n,attack=remove_bool, fdict=fvals,pdict=pvals,lcc_method_relS="pmult",
                                       executable_path = r"C:\Users\jj\Downloads\GitHub\small-perc\libs\p-recursion.exe")
                return (averaged_data, fin)
            #biadjacency matrix
        elif len(content[0]) > 4 or (len(content[0]) == 4 and len(content) == 2):
            print("biadj file")
            if nodecount_bi(file_name) > 100:
                file.close()
                averaged_data = "None"
                fin = "None"
            else:
                edge_list = np.empty(len(content), dtype=object)
                # edge1 = content[0].strip()
                # edge1 = edge1.split("\t")
                k = len(content[0].strip().split("\t"))
                n = len(content) + k
                adj = np.zeros((len(content), k))
                big_adj = np.zeros((n, n))

                for i in range(len(content)):
                    edge = content[i].strip()
                    edge = edge.split("\t")
                    #print("edge")
                    #print(edge)

                    for j in range(len(edge)):
                        e = int(float(edge[j]))
                        if e >= 1:
                            adj[i, j] = 1
                            big_adj[i, j + len(content)] = 1
                            big_adj[j + len(content), i] = 1
                        else:
                            adj[i, j] = 0
                G_0 = nx.from_numpy_array(big_adj)
                # G_0 = nx.algorithms.bipartite.matrix.from_biadjacency_matrix(adj, create_using=None)
                G = G_0.copy()
                nx.draw(G)

                p = len(edge_list) / scipy.special.comb(n, 2)

                x_array = np.arange(0, n) / n
                averaged_data = np.zeros(n)
                print("average bi")
                for j_2 in range(100):
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
                #print(x_array, "xarray")
                averaged_data /= 100
                #print(averaged_data, "y")

                fin = finiteTheory.relSCurve(p,n,attack=remove_bool, fdict=fvals,pdict=pvals,lcc_method_relS="pmult",
                                       executable_path = r"C:\Users\jj\Downloads\GitHub\small-perc\libs\p-recursion.exe")
                
                file.close()

                return (averaged_data, fin)



bottom20_i = [   7,   14,   15,   18,   19,   20,   21,   32,   36,   39,   41,   42,   55,  118,
  164,  183,  193,  207,  214,  218,  230,  234,  240,  244,  250,  251,  260,  264,
  265,  266,  267,  268,  269,  301,  302,  308,  311,  312,  313,  318,  323,  328,
  332,  333,  339,  341,  342,  558,  859,  883,  892,  893,  894,  895,  896,  897,
  899,  900,  901,  909,  918,  919,  922,  923,  924,  956,  957,  958,  959,  960,
  961,  973 , 974,  975,  976,  977,  978,  980,  981,  984,  985,  986,  987,  988,
  990,  992,  993,  994,  997,  998, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1012,
 1015, 1020, 1021, 1022, 1024, 1026, 1027, 1030, 1031, 1032, 1037, 1038, 1039, 1040,
 1041, 1049, 1051, 1055, 1056, 1057, 1058, 1059, 1061, 1062, 1064, 1065, 1074, 1077,
 1078, 1079, 1080, 1081, 1086, 1094, 1138, 1148, 1149, 1150, 1151, 1155, 1158, 1161,
 1163, 1164, 1166, 1167, 1169, 1174, 1176, 1183, 1184, 1185, 1187, 1188, 1195, 1200,
 1201, 1211, 1218, 1220, 1223, 1224, 1226, 1227, 1228, 1243, 1244, 1245, 1249, 1250,
 1251, 1252 ,1253, 1255, 1261, 1262, 1263, 1264, 1265, 1267, 1268, 1269, 1271, 1273,
 1276, 1278, 1279, 1280, 1281, 1282, 1283, 1284, 1285, 1289, 1290, 1291, 1292, 1293,
 1294, 1295, 1296, 1297, 1298, 1299, 1300, 1301, 1302, 1303, 1304, 1305, 1306, 1307,
 1308, 1309, 1310, 1311, 1312, 1313, 1314, 1315, 1316, 1318, 1319, 1320, 1321, 1322,
 1323, 1324, 1325, 1327, 1328, 1329, 1330, 1331, 1332, 1333, 1334, 1336, 1339, 1340,
 1343, 1344, 1345, 1346, 1347, 1348, 1349, 1350, 1351, 1354, 1355, 1356, 1357, 1358,
 1359, 1360, 1361, 1362, 1366, 1368, 1372, 1373, 1374, 1379, 1380, 1382, 1384, 1385,
 1386, 1387, 1388, 1389, 1392, 1393, 1394, 1395, 1401, 1402, 1404, 1405, 1406, 1407,
 1408, 1409, 1410, 1411, 1412, 1413, 1414, 1415, 1416, 1417, 1418, 1421, 1422, 1424,
 1427, 1428, 1429, 1430, 1431, 1432, 1433, 1436, 1437, 1438, 1439, 1440, 1441, 1529,
 1531, 1533, 1535, 1537, 1539, 1541, 1548, 1549, 1555, 1560, 1563, 1569, 1570, 1573,
 1594, 1606]


top20_i =[   8,   17,   25,   26,   27,   28,   29,   30,   37,   59,   62,   63,   89,  112,
  145,  166,  237,  290,  294,  295,  309,  322,  346,  347,  355,  356,  359,  360,
  361,  363,  371,  373,  374,  379,  381,  382,  385,  386,  387,  388,  390,  391,
  392,  393,  394,  395,  396,  397,  399,  402,  404,  405,  406,  407,  408,  409,
  411,  412,  413,  414,  415,  416,  417,  419,  422,  423,  424,  425,  426,  427,
  428,  430,  431,  432,  433,  443,  445,  448,  449,  450,  451,  454,  455,  457,
  459,  461,  462,  464,  466,  468,  472,  473,  474,  478,  489,  492,  495,  496,
  497,  498,  499,  502,  503,  507,  511,  513,  516,  520,  521,  523,  525,  528,
  530,  532,  570,  572,  582,  646,  647,  648,  651,  653,  654,  655,  656,  658,
  660,  661,  662,  663,  665,  666,  667,  669,  670,  671,  672,  681,  682,  684,
  685,  686,  690,  692,  693,  694,  695,  696,  697,  699,  701,  703,  704,  705,
  706,  714,  715,  716,  717,  719,  720,  721,  722,  723,  724,  732,  738,  739,
  741,  742,  746,  748,  750,  751,  752,  758,  759,  760,  764,  770,  771,  772,
  773,  775,  776,  777,  778,  779,  780,  790,  791,  793,  794,  795,  802,  803,
  804,  808,  809,  810,  811,  814,  815,  816,  817,  819,  821,  822,  824,  825,
  826,  827,  828,  831,  832 , 833,  839,  842,  843,  844,  846,  848,  849,  852,
  853,  854,  855,  856,  857,  860,  861,  864,  866,  867,  869,  870,  871,  872,
  873,  874,  876,  877,  881,  882,  884,  885,  886,  887,  888,  889,  891,  938,
  940,  942,  948,  952,  953,  954, 1063, 1096, 1097, 1098, 1099, 1105, 1106, 1108,
 1113, 1114, 1115, 1116, 1118, 1121, 1122, 1124, 1125, 1127, 1128, 1132, 1133, 1135,
 1136, 1139, 1213, 1216, 1221, 1222, 1232, 1258, 1270, 1452, 1453, 1460, 1463, 1470,
 1483, 1499, 1500, 1501, 1502, 1503, 1504, 1506, 1507, 1509, 1510, 1512, 1513, 1515,
 1518, 1519, 1520, 1521, 1522, 1526, 1543, 1544, 1561, 1562, 1582, 1583, 1584, 1585,
 1586, 1587]
 
y_array_b = np.zeros(len(bottom20_i))
y_array_t = np.zeros(len(top20_i))

import os

indices = pd.read_pickle("bayesian indices")
# print("indices")
# print(indices)
# print(indices[0])
# print('iloc')
# print(indices.iloc[0][0])

# for i_b in range(len(bottom20_i)):
#     new_ib = indices.iloc[bottom20_i[i_b]][0]
#     #print("file1",nwks_list2.iloc[i_b][1])
#     file = nwks_list2.iloc[new_ib][1]
#     if nodecount_edge(file_name = file) <= 30:
#     #file2 = file2.replace("C:\\Users\\jj\Downloads\\GitHub\small-perc\\pholme_networks", '')
#     #print("file2",file)
#     #file = file[60:]
#     #print("file3",file2)
#       values = mega_file_reader2(removal = "random", adj_list = [file])
#       print(file)
#       print("val",values)
#       print("averaged data",values[0])
#       print("fin", values[1])
#       y = ((values[0]- values[1]) ** 2).mean()
#       y_array_b[i_b] = y

# y_array_bdf = pd.DataFrame(y_array_b)
# y_array_bdf.to_pickle("bayesian mse bottom 20")

# for i_t in range(len(top20_i)):
#     new_it = indices.iloc[top20_i[i_t]][0]
#     file = nwks_list2.iloc[new_it][1]
#     if nodecount_edge(file_name = file) <= 30:
#       values = mega_file_reader2(removal = "random", adj_list = [file])
#       y = ((values[0]- values[1]) ** 2).mean()
#       y_array_t[i_t] = y

# y_array_tdf = pd.DataFrame(y_array_t)
# y_array_tdf.to_pickle("bayesian mse top 20")

top_indices = pd.read_pickle("bayesian mse top 20")
bottom_indices = pd.read_pickle("bayesian mse bottom 20")


top20x1 = [ 45.,  28.,  40.,  62.,  67.,  49.,  50.,  50.,  49.,  71.,  75.,  74.,   4.,  40.,
  25.,   2.,   2.,  32.,  30.,  36.,   5.,   5.,  37.,  48.,  76.,  91., 100.,  91.,
  88.,  91.,  71.,  77.,  69.,  72.,  52.,  52.,  49.,  50.,  53.,  48.,  94., 100.,
  98.,  95.,  92.,  93.,  95.,  93.,  83.,  80.,  78.,  77.,  78.,  77.,  55.,  57.,
  55.,  52.,  56.,  53.,  55.,  54.,  55.,  99.,  96., 100.,  87.,  65.,  49.,  47.,
  51.,  51.,  46.,  50.,  33.,  33.,  31.,  35.,  34.,  80.,  61.,  61., 100.,  88.,
  84.,  84.,  86.,  79.,  80.,  35.,  36.,  24.,  30.,  30.,  28.,  29.,  25.,  29.,
  29.,  53.,  32.,  32.,  23.,  27.,  28.,  59.,  23.,  30.,  40.,  53.,  43.,  31.,
  38.,  61.,  83.,  89.,  74.,  77.,  70.,  70.,  82.,  83.,  59.,  73.,  64.,  67.,
  56.,  47.,  50.,  51.,  54.,  76.,  83.,  76.,  45.,  42.,  30.,  26.,  40.,  31.,
  41.,  62.,  77.,  85.,  60.,  59.,  51.,  96.,  58.,  85.,  71.,  42.,  58.,  75.,
  46.,  37.,  40.,  22.,  34.,  55.,  52.,  28.,  49.,  49.,  30.,  48.,  32.,  31.,
  45.,  32.,  39.,  68.,  84.,  69.,  42.,  42.,  54.,  47.,  62.,  58.,  72.,  80.,
  83.,  49.,  74.,  59.,  63.,  26.,  40.,  59.,  56.,  57.,  62.,  49.,  48.,  78.,
  92.,  90.,  84.,  64.,  71.,  62.,  57.,  40.,  79.,  82.,  62.,  56.,  55.,  46.,
  81.,  83.,  74.,  69.,  44.,  75.,  15.,  25.,  32.,  32.,  38.,  34.,  71.,  21.,
  18.,  16.,  49.,  44.,  35.,  18.,  24.,  23.,  23.,  32.,  25.,  11.,  60.,  54.,
  68.,  33.,  53.,  44.,  65.,  38.,  20.,  89.,  72.,  71.,  69.,  24.,  28.,  30.,
  29.,  35.,  27.,  30.,  37.,  74.,  75.,  73.,  95., 101.,  99.,  90.,  90.,  80.,
 101.,  92.,  21.,  21.,  21.,  21.,  17.,  14.,  49.,  75.,  75.,  55.,  54.,  92.,
  60.,  62.,  37.,  56.,  75.,  80.,  97.,  51.,  54.,  47.,  61.,  55.,  75.,  41.,
  43.,  43.,  78.,  73.,  80.,  21.,  85.,  62.,  78.,  78.,  45.,  43.,  32.,  55.,
  38.,  24.]

top20x = list(filter(lambda x : x <= 30, top20x1))

bottom20x1 = [ 16.,  75.,  34.,  92.,  92.,  64.,  43.,  28.,  86.,  43.,  34.,  27.,  44.,  28.,
  16.,  15.,  25.,  13.,  48.,  38.,  45.,  23.,  16.,  27.,  23.,  32.,  63.,  59.,
  46.,  68.,  71.,  78.,  52.,  39.,  31.,  18.,  15.,  24.,  42.,  34.,  42.,  34.,
  39.,  31.,  18.,  15.,  24.,  34.,  27.,  20.,  31.,  41.,  33.,  55.,  70.,  44.,
  16.,  18.,  10.,  17.,  29.,  19.,  12.,  12.,   8.,  17.,  22.,  23.,  23.,  27.,
  13.,  19.,  19.,  22.,  23.,  25.,  25.,  27.,  15.,  21.,  32.,  32.,  25.,  23.,
  33.,  22.,  23.,  29.,  16.,  13.,  30.,  29.,  28.,  25.,  24.,  24.,  23.,  38.,
  67.,  17.,  54.,  45.,  33.,  39.,  74.,  16.,  26.,  39.,  15.,  16.,  16.,  14.,
   8.,  23.,  44.,  72.,  76.,  84.,  87.,  86.,  87.,  51.,  15.,  15.,  35.,  30.,
  39.,  45.,  39.,  28.,  45.,  31.,  52.,  33.,  45.,  33.,  13.,  16.,  26.,  47.,
  19.,  31.,  24.,  24.,  24.,  58.,  34.,  32.,  32.,  32.,  28.,  28.,  92.,  39.,
  39.,  21.,  21.,  21.,  21.,  21.,  71.,  71.,  71.,  29.,  29.,  29.,  24.,  24.,
  24.,  24.,  24.,  14.,  86.,  88.,  70.,  12.,  21.,  12.,  16.,  63.,  63.,  24.,
  20.,  20.,  68.,  30.,  27.,  12.,  32.,  26.,  78.,  26.,  44.,  20.,  20.,  90.,
  50.,  56.,  56.,  56.,  62.,  72.,  74.,  78.,  80.,  86.,  96.,  96.,  98.,  98.,
  30.,  12.,  12.,  50.,  57.,  56.,  40.,  56.,  11.,  56.,  56.,  54.,  48.,  72.,
  42.,  44.,  46.,  65., 100.,  45.,  52.,  70.,  70.,  70.,  70.,  14.,  36., 100.,
  16., 100., 100.,  50.,  50.,  42.,  92.,  12.,  30.,  32.,  22.,  22.,  77.,  24.,
  24.,  70.,  16.,  24.,  12.,  18.,  57.,  10.,  11.,  19.,  24.,  27.,  16.,  18.,
  60.,  24.,  60.,  60.,  72.,  36.,  50.,  60.,  94.,  12.,  32.,  84.,  60.,  90.,
  60.,  90.,  18.,  12.,  18.,  24.,  60.,  24.,  46.,  30.,  12.,  50.,  32.,  30.,
  39.,  37.,  37.,  37.,  24.,  24.,  69.,  24.,  39.,  46.,  35.,  20.,  54.,   9.,
  16.,  25.,  36.,  49.,  64.,  81.,  72.,  66.,  84.,  68.,   9.,  66., 100.,  29.,
  27.,  25.]

bottom20x = list(filter(lambda x : x <= 30, bottom20x1))

plt.plot(top20x,y_array_t, marker = "x", color = "red")
plt.plot(bottom20x,y_array_b, marker = "x", color = "green")