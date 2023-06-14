import numpy as np
import scipy
import scipy.stats as sst
import networkx as nx
from random import choice
from scipy.special import comb
import matplotlib.pyplot as plt

def probonly(n,m):
    A = 1 / scipy.special.comb(scipy.special.comb(n, 2), m) * n * scipy.special.comb(n - 1, m)
    return A


def prob_larger(n,m):
    probsum = 0
    for i in range(m, 0, -1):
        if m==1:
            return 1
        else:
            probsum += probonly(n,i)
    return probsum


def probs(n,m):
    probsum = 0
    max = 0
    x_calc = np.zeros(m+1)
    y_calc = np.zeros(m+1)
    for i in range(m,0,-1):
        if i == m:
            if m == 1:
                prob=1

            else:
                prob = 1/scipy.special.comb(scipy.special.comb(n,2),m) * n * scipy.special.comb(n-1,m)*\
                       scipy.special.comb(scipy.special.comb(n,2)-n+1,m-i)/scipy.special.comb(m,i)
            max += prob * m
            probsum += prob
            x_calc[i] = m
            y_calc[i] = max
            print(str(i) + "," + str(prob))
        else:
            prob = n*scipy.special.comb(n-1,i)/scipy.special.comb(scipy.special.comb(n,2),m)*\
                   scipy.special.comb(scipy.special.comb(n,2)-n+1,m-i)/scipy.special.comb(m,i)*(1-prob_larger(n-1,m-i))
            max += i * prob
            probsum += prob
            x_calc[i] = i
            y_calc[i] = max
            print(str(i) + ","+ str(prob))
    print("probsum" + str(probsum))
    return x_calc,y_calc


def sim_attack(n,m):
    x_sim = np.zeros(m+1)
    y_sim = np.zeros(m+1)
    for i in range(m+1):
        G=nx.gnm_random_graph(n,i)
        max = sorted(G.degree, key=lambda x: x[1], reverse=True)[0][1]
        x_sim[i] = i
        y_sim[i] = max
    return x_sim, y_sim


def graph(n,m):
    prob = probs(n,m)
    print(prob[1])
    sim = sim_attack(n,m)
    plt.plot(prob[0], prob[1], label="recursion")
    plt.plot(sim[0],sim[1],label = "simulated")
    plt.legend()
    plt.show()

graph(10,4) #all the max values end up less than 0

#graph(10,40) #gives a lot of 0s for the recursion