###############################################################################
#
# Library of functions to calculate theoretical percolation results for 
# infinite networks.
#
# This library contains the following functions:
#     myLambertW (previously "my_lambertw")
#     relSCurve (previously "perf_sim2")
#     perf_sim2copy (previously "perf_sim2copy")
#     relSmallSCurve (previously "s_sim")
#
# This library previously contained:
#     perf_sim2_attack -> now merged into relSCurve with attack=True
#     s_sim_attack -> now merged into relSmallSCurve with attack=True
#
###############################################################################

import numpy as np
from scipy.special import comb, lambertw
from utils import *
from performanceMeasures import *
#from data import *

def myLambertW(x, k=0, tol=1E-20):
    '''Lambert-W function with interpolation close to the jump point of its 
    zero-th branch. (Using the scipy implementation sometimes does not return
    a number if evaluated too close to the jump point.)

    Parameters
    ----------
    x : float
       Argument of the Lambert-W function.

    k : int (default=0)
       Branch of the Lambert-W function.

    Returns
    -------
    lw : float
       Value of the Lambert-W function (with interpolation near jump point)
    '''

    if np.abs(x + 1 / np.exp(1)) < tol:
        # if input is close to percolation threshold, set output to -1.0
        lw = -1.0
    else:
        lw = lambertw(x, k=k)

    return lw


def relSCurve(n, p, attack=False, reverse=False, smooth_end=False):
    '''Sequence of the expected relative sizes of the largest connected 
    component of an Erdos--Renyi random graph with `n` nodes and edge 
    probability `p` when removing nodes sequentially, either uniformly at
    random or (adaptively) targeted by degree.

    Results are from equations for percolation in the large-n limit.

    Parameters
    ----------
    n : int
       Number of nodes in a graph.
       
    p : float
       Edge probability in a graph.

    attack : bool (default=False)
       If attack is True, target nodes by degree instead of uniformly at 
       random.
       
    reverse : bool (default=False)
       If reverse is True, return expected sizes in reverse order.

    smooth_end : bool (default=False)
       If smooth_end is True, add inverse of current network size as lower 
       bound for expected relative size of the largest connected component.

    Returns
    -------
    relS : 1D numpy array
       Sequence of the expected relative sizes of the largest connected 
       component under sequential node removal.
    '''

    # initialize S array
    relS = np.zeros(n)

    # initialize current_n and current_p
    current_n = n
    current_p = p # could probably delete this now since p doesn't change


    for i in range(n):
        # compute mean degree
        c = 2 * current_p * comb(current_n, 2) / current_n

        # compute value of S from percolation theory for infinite networks
        if c == 1:
            relS[i] = 2/current_n
            # print("set to 2/i")
            # print("currentp",current_p)
            # print("currentn",current_n)
            # print("c",c)
        elif c > 0:
            relS[i] = 1 + np.real(
                myLambertW((-c * np.exp(-c)), k=0, tol=1e-8) / c)
            # print("set to 0")
            # print("currentp",current_p)
            # print("currentn",current_n)
            # print("c",c)
        else:
            relS[i] = 0
            # print("set to 0")
            # print("currentp",current_p)
            # print("currentn",current_n)
            # print("c",c)
        if smooth_end == True:
            relS[i] = max([relS[i], 1 / current_n])

        # update current_p
        if attack:
            current_p = edgeProbabilityAfterTargetedAttack(current_n,current_p)
        else:
            current_p = current_p

        # update current_n
        if current_n > 1:
            current_n -= 1
    #print("relS", relS)
    return relS


def perf_sim2copy(n, p, smooth_end=False):
    '''Only here for debugging purposes.'''
    #this was a function for troubleshooting lambert stuff
    # we did -2 to avoid the end tail bit at first
    y_array = np.zeros(n - 2) # just the part that goes into the lambert function
    z_array = np.zeros(n - 2) # actual S from entire lambert calculation
    x_array = np.zeros(n - 2) # proportion of nodes "removed"
    mean_array = np.zeros(n) #for c values
    difference = np.zeros(n) 
    #for difference between y and 1/e - made sure some values aligned, but I 
    # can't quite remember why - I don't think we use this much anymore though

    new_p = p
    new_n = n
    c_init = 2 * new_p * comb(new_n, 2) / new_n
    percolation_threshold2 = 1 / n + (n - 1) / (c_init * n)

    for i in range(n - 2):
        x_array[i] = i / n

    for i in range(n - 2):
        c = 2 * new_p * comb(new_n, 2) / new_n
        if smoothi_end == True:
            y_array[i] = -c * np.exp(-c)
            z_array[i] = 1 + myLambertW(-c * np.exp(-c)) / c + 1 / new_n
        else:
            y_array[i] = -c * np.exp(-c)
            z_array[i] = 1 + myLambertW(-c * np.exp(-c)) / c
            mean_array[i] = c
            difference[i] = y_array[i] - 1 / scipy.e
        if new_n ** 2 - 2 * new_n == 0:
            new_p = new_p
        else:
            new_p = new_p
            new_n -= 1

    return x_array, y_array, z_array, mean_array, difference, percolation_threshold2


def relSmallSCurve(n, p, attack=False, smoothing=False):
    '''Sequence of the expected mean sizes of the small connected components of
    an Erdos--Renyi random graph with `n` nodes and edge probability `p` when
    removing nodes sequentially, either uniformly at random or (adaptively) 
    targeted by degree.

    Results are from equations for percolation in the large-n limit.

    Parameters
    ----------
    n : list
       Numbers of nodes in a graph.
       
    p : float
       Edge probability in a graph.

    attack : bool (default=False)
       If attack is True, target nodes by degree instead of uniformly at 
       random.
       
    reverse : bool (default=False)
       If reverse is True, return expected sizes in reverse order.
       
    smooth_end : bool (default=False)
       If smooth_end is True, add inverse of current network size as lower 
       bound for expected relative size of the largest connected component.

    Returns
    -------
    rel_s : 1D numpy array
       Sequence of the expected mean sizes of the small connected components
       under sequential node removal.
    '''
    # get fraction of nodes "removed"
    removed_fraction = np.arange(n) / n

    # initialize S array
    rel_s = np.zeros(n)

    # initialize current_n and current_p
    current_n = n
    current_p = p # could probably delete this now since p doesn't change


    for i in range(n):
        # compute mean degree
        c = 2 * current_p * comb(current_n, 2) / current_n
        # compute value of S from percolation theory for infinite networks
        S = 1 + myLambertW((-c * np.exp(-c)), k=0, tol=1e-8) / c
        # compute size of mean small component size
        rel_s[i] = 1 / (1 - c + c * S)

        if smooth_end == True:
            relS[i] = max([relS[i], 1 / new_n])

        # update current_p
        if attack:
            current_p = edgeProbabilityAfterTargetedAttack(current_n,current_p)
        else:     
            current_p = current_p

        # update current_n
        if current_n > 1:
            current_n -= 1

    return removed_fraction, rel_s