import numpy as np
import scipy
import scipy.stats as sst
import networkx as nx
from random import choice
from scipy.special import comb
import matplotlib.pyplot as plt

import numpy as np
from scipy.special import comb as C


def how_many_removal_options(n, m, k):
    each_nodes_edges_at_start = n - 1
    edges_to_remove = C(n, 2, exact=True) - m
    safe_to_remove_from_any = min(n - 1 - k, edges_to_remove)

    initial_removal_options = C(n, safe_to_remove_from_any, exact=True)

    edges_left_to_remove = edges_to_remove - safe_to_remove_from_any

    # This next part is tricky...
    secondary_removal_options = []
    for were_removed_from_key in range(0, safe_to_remove_from_any + 1):
        were_removed_from_other = safe_to_remove_from_any - were_removed_from_key

        probability = C(n - 1, were_removed_from_key, exact=True) / initial_removal_options
        options = C(C(n - 1, 2, exact=True) - were_removed_from_other, edges_left_to_remove, exact=True)

        secondary_removal_options.append(probability * options)
    return initial_removal_options * np.sum(secondary_removal_options)


def probability_of_atleast_degree(n, m, k):
    ways = how_many_removal_options(n, m, k)
    all_combos = C(C(n, 2, exact=True), m, exact=True)

    for_single = ways / all_combos

    if for_single > 1 / n:
        return 1.0
    else:
        return n * for_single

from scipy.special import comb

def probonly(n,m):
    A = 1 / scipy.special.comb(scipy.special.comb(n, 2), m) * n * scipy.special.comb(n - 1, m)
    return A


def prob_larger(n,m,k):
    probsum = 0
    for i in range(m, k, -1):
        if m==1:
            return 1
        else:
            probsum += probonly(n,i)
    return probsum


def probs(n,m):
    probsum = 0
    max = 0
    prob_arr = np.zeros(m+1)
    x_calc = np.zeros(m+1)
    y_calc = np.zeros(m+1)
    for i in range(m,0,-1):
        if i == m:
            if m == 1:
                prob=1

            else:
                prob = n*scipy.special.comb((n-1),m)/scipy.special.comb(scipy.special.comb(n,2),m)
                #prob = 1/scipy.special.comb(scipy.special.comb(n,2),m) * n * scipy.special.comb(n-1,m)\
                       #*scipy.special.comb((scipy.special.comb(n,2)-n+1),m-i)/scipy.special.comb(m,i)
            #max += prob * m
            probsum += prob
            prob_arr[i] = prob
            x_calc[i] = m
            y_calc[i] = max
            print(str(i) + "," + str(prob))
        else:
            prob = n*scipy.special.comb(n-1,i)/scipy.special.comb(scipy.special.comb(n,2),m)*\
                   scipy.special.comb(scipy.special.comb(n,2)-n+1,m-i)/scipy.special.comb(m,i)*(1-probability_of_atleast_degree(n-1,m-i,i))
            max += i * prob
            probsum += prob
            prob_arr[i] = prob
            x_calc[i] = i
            #y_calc[i] = max
            print(str(i) + ","+ str(prob))
    for j in range(m+1):
        max += prob_arr[j]*j
        y_calc[j] = max
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

    plt.xlabel("m")
    plt.ylabel("max")
    return x_sim, y_sim


def graph(n,m):
    prob = probs(n,m)
    print(prob[1])
    sim = sim_attack(n,m)
    plt.plot(prob[0], prob[1], label="recursion")
    plt.plot(sim[0],sim[1],label = "simulated")
    plt.legend()
    plt.show()



graph(10,20) #all the max values end up less than 0

#graph(10,40) #gives a lot of 0s for the recursion