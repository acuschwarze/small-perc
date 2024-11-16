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
#     calcA
#     calcB
#     calcC
#     abc
#     c_graph
#     normalized
#     normalized_table
#     raw_S
#     calculate_S
#     SCurve (previously "big_S")
#     relSCurve (previously "big_relS")
#     SPoints (previously "S_calc_data")
#
# This library previously contained:
#     S_attack -> now merged into SCurve with attack=True
#
###############################################################################

import numpy as np
import scipy.special
from scipy.special import comb, factorial
from libs.utils import *
import matplotlib.pyplot as plt
import math, os
#import cppimport.import_hook
#import recursion

#from glob import glob
#from setuptools import setup
#from pybind11.setup_helpers import Pybind11Extension

import subprocess

def execute_executable(executable_path):
    #print('EE path', executable_path)
    try:
        # Run the executable and capture its output
        result = subprocess.run(executable_path, capture_output=True, text=True, check=True)
        
        # Extract the output
        output = result.stdout.strip()
        #print("EE Output", output)
        #print("EE Output", output, "path", executable_path)
        return output
    except subprocess.CalledProcessError as e:
        # Handle if the executable returns a non-zero exit code
        print("Error: Executable returned non-zero exit code.")
        return None
    except FileNotFoundError:
        # Handle if the executable file is not found
        print("Error: Executable file not found.")
        return None

def raw_f(p, i, n):
    '''Compute f (i.e., the probability that a subgraph with `i` nodes of an 
    Erdos--Renyi random graph with `n` nodes and edge probability `p` is 
    connected.

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
       random graph with `n` nodes and edge probability `p` is connected.
    '''

    if i == 1:
        p_connect = 1

    else:
        sum_f = 0
        for i_n in range(1, i, 1):
            sum_f += (raw_f(p, i_n, n)
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
       random graph with `n` nodes and edge probability `p` is connected.
    '''
    # look for precomputed data
    if p in fdict:
        if n in fdict[p]:
            if i in fdict[p][n]:
                return fdict[p][n][i]

    sum_f = 0
    for i_n in range(1, i, 1):
        sum_f += (calculate_f(p, i_n, n, fdict=fdict)
            * comb(i - 1, i_n - 1) * (1 - p) ** ((i_n) * (i - i_n)))
    f = 1 - sum_f

    return f


def calculate_g(p, i, n):
    '''Compute g (i.e., the probability a selected set of `i` nodes 
    Erdos--Renyi random graph with `n` nodes and edge probability `p` has no
    neighbors in the remaining n-i nodes.

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
       The the probability a selected set of `i` nodes Erdos--Renyi random
       graph with `n` nodes and edge probability `p` has no neighbors in the 
       remaining n-i nodes.
    '''
    g = (1 - p) ** (i * (n - i))

    return g
 

def raw_P(p, i, n):
    '''Compute P (i.e., the probability that an Erdos--Renyi random graph with 
    `n` nodes and edge probability `p` has a largest connected component of 
    size `i`.

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
       The probability that an Erdos--Renyi random graph with `n` nodes and 
       edge probability `p` has a largest connected component of size `i`.
    '''

    if i == 1 and n == 1:
        P_tot = 1
    elif i == 1 and n != 1:
        P_tot = (1 - p) ** comb(n, 2)

    else:
        sum_P = 0
        for j in range(0, i + 1, 1):
            if j==i:
                sum_P += .5*raw_P(p, j, n - i)
            else:
                sum_P += raw_P(p, j, n - i)
        P_tot = comb(n, i) * raw_f(p, i, n) * calculate_g(p, i, n) * sum_P

    return P_tot


def calculate_P(p, i, n, fdict={}, pdict={}): # find P with dictionary
    '''Load or compute P (i.e., the probability that an Erdos--Renyi random 
    graph with `n` nodes and edge probability `p` has a largest connected 
    component of size `i`.

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
       The probability that an Erdos--Renyi random graph with `n` nodes and
       edge probability `p` has a largest connected component of size `i`.
    '''
    if p in pdict:
        if n in pdict[p]:
            if i in pdict[p][n]:
                return pdict[p][n][i]

    if i == 1 and n == 1:
        P_tot = 1
    elif i == 1 and n != 1:
        P_tot = (1 - p) ** comb(n, 2)

    else:
        sum_P = 0

        for j in range(1, i + 1, 1):
            if j==i:
                sum_P += .5 * calculate_P(p, j, n - i, fdict=fdict, pdict=pdict)
            else:
                sum_P += calculate_P(p, j, n - i, fdict=fdict, pdict=pdict)

        P_tot = (scipy.special.comb(n,i)*calculate_f(p, i, n, fdict=fdict)
             * calculate_g(p, i, n) * sum_P) # * factor of ceiling(n/2)??

    return P_tot



def alice_helper(p,i,n,k,fdict={},pdict={}):
    # (n - k*i choose i) * f * g(p,i,n-k*i)
    return scipy.special.comb((n-(k-1)*i),i)*calculate_f(p, i, n, fdict=fdict)* calculate_g(p, i, (n-(k-1)*i))

def alice(p,i,n,fdict={},pdict={}):
    if i == 1 and n == 1:
        P_tot = 1
    elif i == 1 and n != 1:
        P_tot = (1 - p) ** comb(n, 2)

    else:
        P_tot = 0
        for k in range(1,n//i + 1): # each exact number of lcc's to calculate P for

            # find (n choose i)* f * g *(n-i choose i) * f * g etc...
            product = 1
            for k_2 in range(1,k+1):
                product *= alice_helper(p,i,n,k_2,fdict=fdict,pdict=pdict)

            # find P(lcc in other n-k*i nodes < i)
            sum_less = 0
            for j in range(1, i, 1):
                sum_less += alice(p, j, n-k*i, fdict=fdict, pdict=pdict)

            P_tot += 1/math.factorial(k) * product * sum_less
            #scipy.special.comb(scipy.special.comb(n, i), k)
    return P_tot


def raw_S(p, n):
    '''Compute the expected size of the largest connected component of
    an Erdos--Renyi random graph with `n` nodes and edge probability `p` using
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
       random graph with `n` nodes and edge probability `p`.
    '''
    S = 0
    for k in range(1, n + 1):
        S += raw_P(p, k, n) * k

    return S

def calculate_P_mult(p, i, n, executable_path="p-recursion.exe"):

    # Path to the executable
    # pwd = os. getcwd()
    #executable_path = "p-recursion.exe" # {} {} {}".format(p, i, n)

    # Execute the executable and capture its output
    # print(os. getcwd())
    output = float(execute_executable([executable_path, str(p), str(i), str(n)]))
    #print("EEO output pmult", output)
    #output = float(output)

    # return
    print("n",n)
    return output


def new_prob_attack(n,p,executable_path =r"C:\Users\jj\Downloads\GitHub\small-perc\max-degree.exe"):
    
    # Path to the executable
    # pwd = os. getcwd()
    #executable_path = "p-recursion.exe" # {} {} {}".format(p, i, n)

    # Execute the executable and capture its output
    # print(os. getcwd())
    output = float(execute_executable([executable_path, str(n), str(p)]))
    print("EEO output attack", output)
    #output = float(output)

    # return
    return output


def calculate_S(p, n, fdict={}, pdict={},lcc_method = "pmult", executable_path='p-recursion.exe'):
    '''Load or compute the expected size of the largest connected component of
    an Erdos--Renyi random graph with `n` nodes and edge probability `p` using
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
       random graph with `n` nodes and edge probability `p`.
    '''
    if lcc_method == "alice":
        S=0
        for m in range(1,n+1):
            S+=m*alice(p,m,n,fdict=pdict,pdict=pdict)
        return S

    elif lcc_method == "pmult":
        S=0
        for m in range(1,n+1):
            S+=m*calculate_P_mult(p,m,n, executable_path=executable_path)
        return S


def SCurve(p, n, attack=False, reverse=False, fdict={}, pdict={}, lcc_method_Scurve="pmult", executable_path='p-recursion.exe', executable2 = r"C:\Users\jj\Downloads\GitHub\small-perc\max-degree.exe"):
    '''Sequence of the expected sizes of the largest connected component of
    an Erdos--Renyi random graph with `n` nodes and edge probability `p` when
    removing nodes sequentially, either uniformly at random or (adaptively) 
    targeted by degree.
    
    Results are from equations for percolation in finite networks.

    ISSUE #1: Does this work properly for more than one element in n?

    Parameters
    ----------
    p : float
       Edge probability in a graph.

    n : list
       Numbers of nodes in a graph.

    attack : bool (default=False)
       If attack is True, target nodes by degree instead of uniformly at 
       random.
       
    reverse : bool (default=False)
       If reverse is True, return expected sizes in reverse order.

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
    
    # initialize array (assume that n has only one entry for now)
    S = np.zeros(n)

    current_p = p
    for i in range(n-1, -1, -1):
        # calculate S for each value <= n
        S[i] = calculate_S(current_p, i+1, fdict=fdict, pdict=pdict, lcc_method=lcc_method_Scurve, executable_path=executable_path)
        #S[i] = raw_S(current_p,i+1)

        if attack:
            # update p only if nodes are removed by degree
            print("run attack")
            current_p = edgeProbabilityAfterTargetedAttack(i+1, current_p) # old code # add plus 1?

    if reverse:
        S = S[::-1]

    return S


def relSCurve(p, n, attack=False, reverse=True, fdict={}, pdict={}, lcc_method_relS = "pmult", executable_path='p-recursion.exe',executable2 = r"C:\Users\jj\Downloads\GitHub\small-perc\max-degree.exe"):
    '''Sequence of the expected relative sizes of the largest connected 
    component of an Erdos--Renyi random graph with `n` nodes and edge 
    probability `p` when removing nodes sequentially, either uniformly at
    random or (adaptively) targeted by degree.

    Results are from equations for percolation in finite networks.

    ISSUE #1: Does this work properly for more than one element in n?

    Parameters
    ----------
    p : float
       Edge probability in a graph.

    n : list
       Numbers of nodes in a graph.

    attack : bool (default=False)
       If attack is True, target nodes by degree instead of uniformly at 
       random.
       
    reverse : bool (default=False)
       If reverse is True, return expected sizes in reverse order.

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

    # assume that n has only one entry for now
    network_sizes = np.arange(1,n+1)
    
    if reverse:
        network_sizes = network_sizes[::-1]

    relS = (SCurve(p, n, attack=attack, reverse=reverse, executable_path=executable_path, executable2=executable2,
        fdict=fdict, pdict=pdict, lcc_method_Scurve = lcc_method_relS) / network_sizes)

    return relS


def SPoints(p=.1, n=[20,50,100], attack=False, reverse=False, 
    fdict={}, pdict={}):
    '''List of the expected sizes of the largest connected component of
    an Erdos--Renyi random graph with `n[i]` nodes and edge probability `p`.
    Results are from equations for percolation in finite networks.

    Parameters
    ----------
    p : float
       Edge probability in a graph.

    n : list
       Numbers of nodes in a graph.
       
    attack : bool (default=False)
       If attack is True, target nodes by degree instead of uniformly at 
       random.
       
    reverse : bool (default=False)
       If reverse is True, return expected sizes in reverse order.

    fdict (default={})
       Dictionary of precomputed values of f.
       
    pdict (default={})
       Dictionary of precomputed values of P.

    Returns
    -------
    n : 1D numpy array
       List of network sizes.

    sizes : 1D numpy array
       List of corresponding expected sizes of the largest connected.
    '''

    sizes = np.array([
        calculate_S(p, nval, attack=attack, reverse=reverse,
            fdict=fdict, pdict=pdict) for nval in n])

    return n, sizes


