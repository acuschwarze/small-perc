###############################################################################
#
# Library of functions to calculate theoretical percolation results for finite
# networks.
#
# This library contains the following functions:
#     raw_f
#     calculate_f
#     calculate_g (previously "g")
#     raw_P
#     calculate_P
#     raw_S
#     calculate_S
#     SCurve (previously "big_S")
#     relSCurve (previously "big_relS ")
#     S_calc_data
#
###############################################################################

import numpy as np
import scipy
import scipy.stats as sst
import networkx as nx
import matplotlib.pyplot as plt
from random import choice
from scipy.special import comb
from data import *
from Dictionaries import *

def raw_f(p, i, n): 
    '''Compute f (i.e., the probability that a subgraph with `i` nodes of an 
    Erdos--Renyi random graph with n nodes and edge probability p is connected.

    Parameters
    ----------
    p : float
       Edge probability in a parent graph.

    i : int
       Number of nodes in a subgraph.
       
    n : int
       Number of nodes in a parent graph.

    Returns
    -------
    p_connect : float
       The probability that a subgraph with `i` nodes of an Erdos--Renyi
       random graph with n nodes and edge probability p is connected.
    '''
    if i == 0:
        p_connect = 0
    if i == 1:
        p_connect = 1
    else:
        sum_f = 0
        for i_n in range(1, i, 1):
            sum_f += (f(p, i_n, n) 
                * comb(i - 1, i_n - 1) * (1 - p) ** ((i_n) * (i - i_n)))
        p_connect = 1 - sum_f

    return p_connect


def calculate_f(p, i, n, fdict={}): # using dictionary to calculate f values
    '''Load or compute f (i.e., the probability that a subgraph with `i` nodes
    of an Erdos--Renyi random graph with n nodes and edge probability p is 
    connected.

    Parameters
    ----------
    p : float
       Edge probability in a parent graph.

    i : int
       Number of nodes in a subgraph.
       
    n : int
       Number of nodes in a parent graph.
       
    fdict (default={})
       Dictionary of precomputed values of f.

    Returns
    -------
    f : float
       The probability that a subgraph with `i` nodes of an Erdos--Renyi
       random graph with n nodes and edge probability p is connected.
    '''
    # look for precomputed data
    if p in fdict:
        if n in fdict[p]:
            if i in fdict[p][n]:
                return fdict[p][n][i]

    # if none is found start computation
    if i == 0:
        f = 0
    if i == 1:
        f = 1
    else:
        sum_f = 0
        for i_n in range(1, i, 1):
            sum_f += (calculate_f(p, i_n, n, fdict=fdict) 
                * comb(i - 1, i_n - 1) * (1 - p) ** ((i_n) * (i - i_n)))
        f = 1 - sum_f

    return f


def calculate_g(p, i, n):
    '''Compute g (i.e., the probability that a subgraph with `i` nodes of an
    Erdos--Renyi random graph with n nodes and edge probability p is connected.
    
    AS: What is g again?

    Parameters
    ----------
    p : float
       Edge probability in a parent graph.

    i : int
       Number of nodes in a subgraph.
       
    n : int
       Number of nodes in a parent graph.

    Returns
    -------
    g : float
       The probability that a subgraph with `i` nodes of an Erdos--Renyi
       random graph with n nodes and edge probability p is connected.
    '''
    g = (1 - p) ** (i * (n - i))

    return g
 

def raw_P(p, i, n):
    '''Compute P (i.e., the probability that a subgraph with `i` nodes of an
    Erdos--Renyi random graph with n nodes and edge probability p is connected.
    
    AS: What is P again?
    
    ISSUE #1: Did we decide on the range of j values?

    Parameters
    ----------
    p : float
       Edge probability in a parent graph.

    i : int
       Number of nodes in a subgraph.
       
    n : int
       Number of nodes in a parent graph.

    Returns
    -------
    P_tot : float
       The probability that a subgraph with `i` nodes of an Erdos--Renyi
       random graph with n nodes and edge probability p is connected.
    '''
    if i == 0 and n == 0:
        P_tot = 1
    elif i > 0 and n == 0:
        P_tot = 0
    elif i > n or n < 0 or i <= 0:
        P_tot = 0
    elif i == 1 and n == 1:
        P_tot = 1
    elif i == 1 and n != 1:
        P_tot = (1 - p) ** comb(n, 2)
    else:
        sum_P = 0
        for j in range(0, i + 1, 1):  # shouldn't it be i+1?
            # AS : Did we ever resolve this issue?
            sum_P += P(p, j, n - i)
        P_tot = comb(n, i) * f(p, i, n) * g(p, i, n) * sum_P
    return P_tot


def calculate_P(p, i, n, fdict={}, pdict={}):
    '''Load or compute P (i.e., the probability that a subgraph with `i` nodes 
    of an Erdos--Renyi random graph with n nodes and edge probability p is 
    connected.
    
    AS: What is P again?
    
    ISSUE #1: Did we decide on the range of j values?

    Parameters
    ----------
    p : float
       Edge probability in a parent graph.

    i : int
       Number of nodes in a subgraph.
       
    n : int
       Number of nodes in a parent graph.

    fdict (default={})
       Dictionary of precomputed values of f.
       
    pdict (default={})
       Dictionary of precomputed values of P.

    Returns
    -------
    P_tot : float
       The probability that a subgraph with `i` nodes of an Erdos--Renyi
       random graph with n nodes and edge probability p is connected.
    '''
    if p in pdict:
        if n in pdict[p]:
            if i in pdict[p][n]:
                return pdict[p][n][i]

    if i == 0 and n == 0:
        P_tot = 1
    elif i > 0 and n == 0:
        P_tot = 0
    elif i > n or n < 0 or i <= 0:
        P_tot = 0
    elif i == 1 and n == 1:
        P_tot = 1
    elif i == 1 and n != 1:
        P_tot = (1 - p) ** comb(n, 2)
    else:
        sum_P = 0
        for j in range(0, i + 1, 1):  # shouldn't it be i+1?
            sum_P += calculate_P(p, j, n - i, fdict=fdict, pdict=pdict)
        P_tot = (comb(n, i)
            * calculate_f(p, i, n, fdict=fdict) * g(p, i, n) * sum_P)

    return P_tot


def raw_S(p, n):
    '''Compute the expected size of the largest connected component of
    an Erdos--Renyi random graph with n nodes and edge probability p using 
    equations for percolation in finite networks.

    Parameters
    ----------
    p : float
       Edge probability in a graph.

    n : int
       Number of nodes in a graph.

    Returns
    -------
    S : float
       Expected size of the largest connected component of an Erdos--Renyi
       random graph with n nodes and edge probability p.
    '''
    S = 0
    for k in range(1, n + 1):
        S += P(p, k, n) * k

    return S


def calculate_S(p, n, fdict={}, pdict={}):
    '''Load or compute the expected size of the largest connected component of 
    an Erdos--Renyi random graph with n nodes and edge probability p using 
    equations for percolation in finite networks.

    Parameters
    ----------
    p : float
       Edge probability in a graph.

    n : int
       Number of nodes in a graph.

    fdict (default={})
       Dictionary of precomputed values of f.
       
    pdict (default={})
       Dictionary of precomputed values of P.

    Returns
    -------
    S : float
       Expected size of the largest connected component of an Erdos--Renyi
       random graph with n nodes and edge probability p.
    '''
    S = 0
    for k in range(1, n + 1):
        S += calculate_P(p, k, n, fdict=fdict, pdict=pdict) * k

    return S

def SCurve(p, n, fdict=fvalues, pdict=pvalues):
    '''Sequence of the expected sizes of the largest connected component of
    an Erdos--Renyi random graph with n nodes and edge probability p when
    removing nodes sequentially uniformly at random. Results are from
    equations for percolation in finite networks.
    
    ISSUE #1: Does this work properly for more than one element in n?

    Parameters
    ----------
    p : float
       Edge probability in a graph.

    n : list
       Numbers of nodes in a graph.

    fdict (default={})
       Dictionary of precomputed values of f.
       
    pdict (default={})
       Dictionary of precomputed values of P.

    Returns
    -------
    S : 1D numpy array
       Sequence of the expected sizes of the largest connected component under
       sequential node removal.
    '''
    #usually i input an array with one n value for now
    S = np.zeros(n[0]) #takes that n value and makes a new array with that length

    for i in range(1,n[0]+1): #indexing purposes
        S[i-1] = calculate_S(p, i, fdict=fdict, pdict=pdict) #finds S for each value <= n

    return S


def big_relS(p, n, fdict={}, pdict={}):
    '''Sequence of the expected relative sizes of the largest connected 
    component of an Erdos--Renyi random graph with n nodes and edge probability 
    p when removing nodes sequentially uniformly at random. Results are from
    equations for percolation in finite networks.
    
    ISSUE #1: Does this work properly for more than one element in n?

    Parameters
    ----------
    p : float
       Edge probability in a graph.

    n : list
       Numbers of nodes in a graph.

    fdict (default={})
       Dictionary of precomputed values of f.
       
    pdict (default={})
       Dictionary of precomputed values of P.

    Returns
    -------
    relS : 1D numpy array
       Sequence of the expected relative sizes of the largest connected 
       component under sequential node removal.
    '''
    #i also input just one n value
    relS = SCurve(p, n, fdict=fdict, pdict=pdict) / np.arange(1,n[0]+1)

    return relS


def S_calc_data(p=.1,n=[20,50,100]): 
    '''What happens here?
    '''
    #returns the values of S for multiple n values.
    x_array = np.zeros(len(n))
    y_array = np.zeros(len(n))

    for a in range(len(n)): 
        # i think this is actually the same as n?
        x_array[a] = n[a]

    for b in range(len(n)):
        y_array[b] = calculate_S(p,n[b])
    print(x_array, y_array)

    return x_array, y_array

def S_attack(p, n, fdict={}, pdict={}):
    S = np.zeros(n[0])  # takes that n value and makes a new array with that length
    new_p = p
    for i in range(n[0], 0, -1):
        S[i - 1] = calculate_S(new_p, i, fdict=fdict, pdict=pdict)  # finds S for each value <= n
        new_p = new_p * i / (i - 2) - 2 * max / ((i - 1) * (i - 2))

    return S