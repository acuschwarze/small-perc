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
    # print('EE path', executable_path)
    try:
        # Run the executable and capture its output
        result = subprocess.run(executable_path, capture_output=True, text=True, check=True)
        
        # Extract the output
        output = result.stdout.strip()
        # rint("EE Output", output)
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
    # if i == 0:
    #     p_connect = 0
    if i == 1:
        p_connect = 1
    # elif i > n:
    #     p_connect = 0
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

    # if none is found start computation
    # if i == 0:
    #     f = 0
    # if i == 1:
    #     f = 1
    # # elif i>n:
    # #     f = 0
    # else:
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
    # if i == 0 and n == 0:
    #     P_tot = 1
    # if i > 0 and n == 0:
    #     P_tot = 0
    # if i > n or n < 0 or i <= 0:
    #     P_tot = 0
    if i == 1 and n == 1:
        P_tot = 1
    elif i == 1 and n != 1:
        P_tot = (1 - p) ** comb(n, 2)
    # elif i == n:
    #     P_tot = raw_f(p,n,n)
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

    # if i == 0 and n == 0:
    #     P_tot = 1
    # if i > 0 and n == 0:
    #     P_tot = 0
    # if i > n or n < 0 or i <= 0:
    #     P_tot = 0
    if i == 1 and n == 1:
        P_tot = 1
    elif i == 1 and n != 1:
        P_tot = (1 - p) ** comb(n, 2)
    # elif i == n:
    #     P_tot = calculate_f(p,n,n)
    else:
        sum_P = 0

        # 1-sum (P(p,j,n)) for j=i+1 to n
        # for j in range(i+1, n+1, 1):
        #     sum_P += calculate_P(p, j, n - i, fdict=fdict, pdict=pdict)
        # P_tot = (scipy.special.comb(n, i) * calculate_f(p, i, n, fdict=fdict)
        #          * calculate_g(p, i, n) * (1 - sum_P))  # * factor of ceiling(n/2)??
        #
        # normal way with j = 1 to i
        for j in range(1, i + 1, 1):
            if j==i:
                sum_P += .5 * calculate_P(p, j, n - i, fdict=fdict, pdict=pdict)
            else:
                sum_P += calculate_P(p, j, n - i, fdict=fdict, pdict=pdict)

        P_tot = (scipy.special.comb(n,i)*calculate_f(p, i, n, fdict=fdict)
             * calculate_g(p, i, n) * sum_P) # * factor of ceiling(n/2)??

    return P_tot


def calcA(p,i,n): # calculate P_tot without j=i
    P_tot = 0
    for j in range(i):
        P_tot+=calculate_P(p,j,n-i)
    P_tot = scipy.special.comb(n,i)*calculate_f(p,i,n)*calculate_g(p,i,n)*P_tot
    return P_tot

def calcB(p,i,n): # calculate P_tot for only j=i
    P_tot = scipy.special.comb(n,i)*calculate_f(p,i,n)*calculate_g(p,i,n)*calculate_P(p,i,n-i)
    return P_tot

def calcC(p,n): # find the factor C such that sum of calcA + calcB * c over all j<=i = 1
    A = 0
    B = 0
    for i_i in range(1,n+1):
        A+=calcA(p,i_i,n)
        B+=calcB(p,i_i,n)
    if B == 0:
        c = 0
    else:
        c = (1 - A) / B
    return c


def abc(p,i,n): # make an array with P(p,i,n) for i<=n
    # use calcA, calB, calcC
  a_table = np.zeros(i+1)
  b_table = np.zeros(i+1)
  abc_table = np.zeros(i+1)

  A = 0
  B = 0
  for i_i in range(1,n+1):
    a_table[i_i] = calcA(p,i_i,n)
    b_table[i_i] = calcB(p,i_i,n)
    A+=a_table[i_i]
    B+=b_table[i_i]
  if B == 0:
    c = 0
  else:
    c = (1-A)/B
  for j in range(1,n+1):
    # find P(p,j,n) with the equation: calcA(p,j,n)+calcC(p,j,n)*calcB(p,j,n) = P(p,j,n)
    abc_table[j] = a_table[j] + c*b_table[j]
  return abc_table


def c_graph(n,p,to_vary): # graph c over n or p for various levels of p or n respectively
    if to_vary == "n":
        for i_p in range(len(p)):
            c_vals = np.zeros(n[0]+1)
            n_vals = np.linspace(0,n[0],n[0]+1)
            for i_n in range(n[0]+1):
                c_vals[i_n] = calcC(p[i_p],i_n)
            plt.plot(n_vals, c_vals, label="p="+str(p[i_p]))
            print("plotted")
        print(n_vals)
        print(c_vals)
        plt.xlabel("n")
        plt.ylabel("c")

    elif to_vary == "p":
        for i_n in range(len(n)):
            print(n[i_n])
            p_vals = np.linspace(0, 1, 81, endpoint=True)
            c_vals = np.zeros(len(p_vals))
            for i_p in range(11):
                c_vals[i_p] = calcC(p_vals[i_p],n[i_n])
            plt.plot(p_vals, c_vals, label="n="+str(n[i_n]))
            print("plotted")
        print(p_vals)
        print(c_vals)
        plt.xlabel("p")
        plt.ylabel("c")
    "should show"
    plt.legend()
    plt.show()


def normalized(p,i,n):
  # calculates P for a certain i and n
  # normalizes the P values during the calculation rather than just at the end

  p_vals = np.zeros(n+1)
  for i_i in range(n+1):
    print(i_i)
    # if i_i == 0 and n == 0:
    #   P_tot = 1
    # if i_i > 0 and n == 0:
    #       P_tot = 0
    # if i_i > n or n < 0 or i_i <= 0:
    #       P_tot = 0
    if i_i == 1 and n != 1:
          P_tot = (1 - p) ** comb(n, 2)
    # elif i_i == n:
    #   P_tot = calculate_f(p,i_i,n)
    else:
      P_tot = 0
      for j in range(1,i_i+1):
          if j==i:
            P_tot+=.5*normalized(p,j,n-i_i)
          else:
            P_tot += normalized(p, j, n - i_i)
      P_tot = scipy.special.comb(n,i_i)*calculate_f(p,i_i,n)*calculate_g(p,i_i,n)*P_tot
    p_vals[i_i] = P_tot
  c = np.sum(p_vals)
  if i<=n:
    return p_vals[i]/c
  else:
    return 0

def normalized_table(p,i,n):
    # creates table of P probabilities that are normalized during the calculations

  p_table = np.zeros(shape=(n+1,i+1))
  for i_n in range(n+1):
    for i_i in range(i+1):
      p_table[i_n,i_i] = normalized(p,i_i,i_n)
  return p_table


def alice_helper(p,i,n,k,fdict={},pdict={}):
    # (n - k*i choose i) * f * g(p,i,n-k*i)
    return scipy.special.comb((n-(k-1)*i),i)*calculate_f(p, i, n, fdict=fdict)* calculate_g(p, i, (n-(k-1)*i))

def alice(p,i,n,fdict={},pdict={}):
    # if i == 0 and n == 0:
    #     P_tot = 1
    # if i > 0 and n == 0:
    #     P_tot = 0
    # if i > n or n < 0 or i <= 0:
    #     P_tot = 0
    if i == 1 and n == 1:
        P_tot = 1
    elif i == 1 and n != 1:
        P_tot = (1 - p) ** comb(n, 2)
    # elif i == n:
    #     P_tot = calculate_f(p,n,n)
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
    #print("EEO output", output)
    #output = float(output)

    # return
    return output


def calculate_S(p, n, fdict={}, pdict={},lcc_method = "abc"):
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
    if lcc_method == "abc":
        S = 0
        p_table = abc(p, n, n)
        for k in range(1, n + 1):
            S += p_table[k] * k
        return S

    elif lcc_method == ".5 factor":
        S = 0
        for k in range(1, n + 1):
            S += k*calculate_P(p,k,n)
            #S += k*raw_P(p,k,n)
        return S

    elif lcc_method == "alice":
        S=0
        for m in range(1,n+1):
            S+=m*alice(p,m,n,fdict=pdict,pdict=pdict)
        return S

    elif lcc_method == "pmult":
        S=0
        for m in range(1,n+1):
            S+=m*calculate_P_mult(p,m,n)
        return S

    elif lcc_method == "normalized during":
        S = 0
        p_table = normalized_table(p, n, n)
        for k in range(1, n + 1):
            S += p_table[n, k] * k
        return S


def SCurve(p, n, attack=False, reverse=False, fdict={}, pdict={}, lcc_method_Scurve="abc"):
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
        S[i] = calculate_S(current_p, i+1, fdict=fdict, pdict=pdict, lcc_method=lcc_method_Scurve)
        #S[i] = raw_S(current_p,i+1)

        if attack:
            # update p only if nodes are removed by degree
            current_p = edgeProbabilityAfterTargetedAttack(i+1, current_p)
            #print(current_p,i+1, S[i])

    if reverse:
        S = S[::-1]

    return S


def relSCurve(p, n, attack=False, reverse=True, fdict={}, pdict={}, lcc_method_relS = "abc"):
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

    relS = (SCurve(p, n, attack=attack, reverse=reverse,
        fdict=fdict, pdict=pdict, lcc_method_Scurve = lcc_method_relS) / network_sizes)

    #print(n,SCurve(p, n, attack=attack, reverse=reverse,
    #    fdict=fdict, pdict=pdict, lcc_method_Scurve = lcc_method_relS))
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


