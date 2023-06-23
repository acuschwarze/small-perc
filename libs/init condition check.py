import numpy as np
import scipy
import scipy.stats as sst
import networkx as nx
from random import choice
from scipy.special import comb
import matplotlib.pyplot as plt

def prob_lcc(n,m):
  degree_count = 0
  for i in range(100):
    G=nx.gnm_random_graph(n,m)
    for node in G.degree:
      if node[1] == m:
        degree_count += 1
  print(degree_count)
  sim_prob = degree_count / (n*100)
  prob = n*scipy.special.comb((n-1),m)/scipy.special.comb(scipy.special.comb(n,2),m)
  print("P(k=m)=" + str(prob))
  print("simulated P(k=m)=" +str(sim_prob))

prob_lcc(10,4)
prob_lcc(10,9)


# if m>n then we have issues with the initial condition so I wanted to get probabilities for k=/= m
def prob_lcc_k(n,m,k):
  degree_count = 0
  for i in range(100):
    G=nx.gnm_random_graph(n,m)
    for node in G.degree:
      if node[1] == k:
        degree_count += 1
  print(degree_count)
  sim_prob = degree_count / (n*100)
  prob = n*scipy.special.comb((n-1),m)/scipy.special.comb(scipy.special.comb(n,2),m)
  print("P(k="+str(k)+")=" + str(prob) + ",(n,m,k)=")
  print("simulated P(k="+str(k)+")=" +str(sim_prob))


prob_lcc_k(15,100,3)
prob_lcc_k(15,50,3)
