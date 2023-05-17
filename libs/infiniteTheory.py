###############################################################################
#
# Library of functions to calculate theoretical percolation results for 
# infinite networks.
#
# This library contains the following functions:
#     perf_sim2 (previously "perf_sim2")
#     my_lambertw
#     perf_sim2copy (previously "perf_sim2copy")
#     perf_sim2_attack (previously "perf_sim2_attack")
#     perf_sim_revision1 (previously "perf_sim_revision1")
#     s_sim
#     s_sim_attack
#
###############################################################################

import numpy as np
import networkx as nx
from scipy.special import comb, lambertw
from scipy.optimize import fsolve
from data import *

def perf_sim2(n, p, smoothing=False):
    # y_array = np.zeros(n-2)
    # x_array = np.zeros(n-2)
    # for i in range(n-2):
    y_array = np.zeros(n)
    x_array = np.zeros(n)
    for i in range(n):
        x_array[i] = i / n #get proportion of nodes "removed"

    new_p = p # could probably delete this now since p doesn't change
    new_n = n

    for i in range(n):
        c = 2 * new_p * comb(new_n, 2) / new_n #mean degree
        if smoothing == True:
            y_array[i] = 1 + lambertw((-c * np.exp(-c)), k=0, tol=1e-8) / c + 1 / new_n
        else:
            y_array[i] = 1 + lambertw((-c * np.exp(-c)), k=0, tol=1e-8) / c
        if new_n ** 2 - 2 * new_n == 0: # can delete this if else statement
            new_p = new_p
        else:
            new_p = new_p
            new_n -= 1

    return x_array, y_array


def my_lambertw(x, k=0): # lambert function with the values near percolation threshold guaranteed (sometimes
                        # just doing lambert function will leave small gaps in the graph)
    if x + 1 / np.exp(1) < 1E-20:
        return -1.0
    else:
        return lambertw(x, k=k)


def perf_sim2copy(n, p, smoothing=False): #this was a function for troubleshooting lambert stuff
    # y_array = np.zeros(n-2)
    # x_array = np.zeros(n-2)
    # for i in range(n-2):
    # we did -2 to avoid the end tail bit at first
    y_array = np.zeros(n - 2) # just the part that goes into the lambert function
    z_array = np.zeros(n - 2) # actual S from entire lambert calculation
    x_array = np.zeros(n - 2) # proportion of nodes "removed"
    mean_array = np.zeros(n) #for c values
    difference = np.zeros(n) #for difference between y and 1/e - made sure some values aligned, but I can't quite
                            #remember why - I don't think we use this much anymore though

    new_p = p
    new_n = n
    c_init = 2 * new_p * comb(new_n, 2) / new_n
    percolation_threshold2 = 1 / n + (n - 1) / (c_init * n)

    for i in range(n - 2):
        x_array[i] = i / n

    for i in range(n - 2):
        c = 2 * new_p * comb(new_n, 2) / new_n
        if smoothing == True:
            y_array[i] = -c * np.exp(-c)
            z_array[i] = 1 + my_lambertw(-c * np.exp(-c)) / c + 1 / new_n
        else:
            y_array[i] = -c * np.exp(-c)
            z_array[i] = 1 + my_lambertw(-c * np.exp(-c)) / c
            mean_array[i] = c
            difference[i] = y_array[i] - 1 / scipy.e
        if new_n ** 2 - 2 * new_n == 0:
            new_p = new_p
        else:
            new_p = new_p
            new_n -= 1

    return x_array, y_array, z_array, mean_array, difference, percolation_threshold2


def perf_sim2_attack(n, p, smoothing=False): #does the lambert calculations but with attacks
    y_array = np.zeros(n)
    x_array = np.zeros(n)
    for i in range(n):
        x_array[i] = i / n

    new_p = p #this time probably can't remove this
    new_n = n

    for i in range(n - 2):
        max = maxdegree(new_n, new_p)
        c = 2 * new_p * comb(new_n, 2) / new_n
        if smoothing == True:
            y_array[i] = 1 + lambertw((-c * np.exp(-c)), k=0, tol=1e-8) / c + 1 / new_n
        else:
            y_array[i] = 1 + lambertw((-c * np.exp(-c)), k=0, tol=1e-8) / c

        if new_n ** 2 - 2 * new_n == 0:
            new_p = new_p
        else:
            new_p = new_p * new_n / (new_n - 2) - 2 * max / ((new_n - 1) * (new_n - 2))
            new_n -= 1

    return x_array, y_array


def perf_sim_revision1(n, p, smoothing=False): #I think this ended up not working well. Could delete
    y_array = np.zeros(n)
    x_array = np.zeros(n)
    for i in range(n):
        x_array[i] = i / n

    new_p = p
    new_n = n
    for i in range(n - 2):
        c = 2 * new_p * comb(new_n, 2) / new_n

        # the function below is from the book - before they approximate using n to infinity
        func = lambda x: (new_n - 1) * np.log(1 - c / (new_n - 1) * (1 - x)) - np.log(x)
        # I don't think this function actually does anything with S
        S = 1 + lambertw((-c * np.exp(-c)), k=0, tol=1e-8) / c
        # solving the function for the rel lcc - but because of logs, there were lots of random jumps
        S2 = scipy.optimize.fsolve(func, np.real(S))
        print(S2)
        y_array[i] = S2
        if new_n ** 2 - 2 * new_n == 0:
            new_p = new_p
        else:
            new_p = new_p * new_n / (new_n - 2) - 2 * c / ((new_n - 1) * (new_n - 2))
            new_n -= 1

    return x_array, y_array


def s_sim(n, p, smoothing=False): # i think this is for little s - the average small component size
    y_array = np.zeros(n)
    x_array = np.zeros(n)
    new_p = p
    new_n = n
    for i in range(n):
        x_array[i] = i / n

    for i in range(n - 2):
        c = 2 * new_p * comb(new_n, 2) / new_n
        # print(c)
        S = 1 + scipy.special.lambertw((-c * np.exp(-c)), k=0, tol=1e-8) / c
        # <s> = 1/(1-c+cS)
        y_array[i] = 1 / (1 - c + c * S)
        # print(y_array[i])
        if new_n ** 2 - 2 * new_n == 0:
            new_p = new_p
        else:
            new_p = new_p * (new_n ** 2 - 2 * new_n + 4) / (new_n ** 2 - 2 * new_n)
        new_n -= 1

    return x_array, y_array


def s_sim_attack(n, p, smoothing=False): # average small component size for attack removals
    y_array = np.zeros(n)
    x_array = np.zeros(n)

    new_p = p
    new_n = n
    for i in range(n):
        x_array[i] = i / n

    for i in range(n - 2):
        max = maxdegree(new_n, new_p)
        c = 2 * new_p * comb(new_n, 2) / new_n
        # print(c)
        S = 1 + lambertw((-c * np.exp(-c)), k=0, tol=1e-8) / c
        # <s> = 1/(1-c+cS)
        y_array[i] = 1 / (1 - c + c * S)
        # print(y_array[i])
        if new_n ** 2 - 2 * new_n == 0:
            new_p = new_p
        else:
            new_p = new_p * new_n / (new_n - 2) - 2 * max / ((new_n - 1) * (new_n - 2))
            # - 2*max/((new_n-1)(new_n-2))
        new_n -= 1

    return x_array, y_array