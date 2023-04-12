# This is a script to generate data for the recursion equation.

# IMPORT PACKAGES
import numpy as np
import scipy
from scipy.special import comb 
from random import choice
import sys, argparse # import packages to manage input arguments for scripts
import os, pickle, csv # import packages for file I/O
import time # package to help keep track of calculation time


# DEFINE FUNCTIONS
def raw_f(p,i,n):
    if i == 0:
        p_connect = 0
    if i == 1:
        p_connect = 1
    else:
        sum_f = 0
        for i_n in range(1,i,1):
            sum_f += f(p,i_n,n)*scipy.special.comb(i-1,i_n-1)*(1-p)**((i_n)*(i-i_n))
        p_connect = 1-sum_f
    return p_connect

def calculate_f(p,i,n, fdict={}):
    
    if p in fdict:
        if n in fdict[p]:
            if i in fdict[p][n]:
                return fdict[p][n][i]
            
    if i == 0:
        p_connect = 0
    if i == 1:
        p_connect = 1
    else:
        sum_f = 0
        for i_n in range(1,i,1):
            sum_f += calculate_f(p,i_n,n, fdict=fdict)*scipy.special.comb(i-1,i_n-1)*(1-p)**((i_n)*(i-i_n))
        p_connect = 1-sum_f
    return p_connect
  
    
def g(p,i,n):
    return (1-p)**(i*(n-i))
  
    
def raw_P(p,i,n):
    if i==0 and n==0:
        P_tot = 1
    elif i>0 and n==0:
        P_tot = 0
    elif i > n or n < 0 or i<=0:
        P_tot = 0
    elif i == 1 and n == 1:
        P_tot = 1
    elif i == 1 and n != 1:
        P_tot = (1-p)**scipy.special.comb(n,2)
    else:
        sum_P = 0
        for j in range(0,i+1,1): # shouldn't it be i+1?
            sum_P += P(p,j,n-i)
        P_tot = scipy.special.comb(n,i)*f(p,i,n)*g(p,i,n)*sum_P
    return P_tot


def calculate_P(p,i,n, fdict={}, pdict={}):
    if i==0 and n==0:
        P_tot = 1
    elif i>0 and n==0:
        P_tot = 0
    elif i > n or n < 0 or i<=0:
        P_tot = 0
    elif i == 1 and n == 1:
        P_tot = 1
    elif i == 1 and n != 1:
        P_tot = (1-p)**scipy.special.comb(n,2)
    else:
        sum_P = 0
        for j in range(0,i+1,1): # shouldn't it be i+1?
            sum_P += calculate_P(p,j,n-i, fdict=fdict, pdict=pdict)
        P_tot = scipy.special.comb(n,i)*calculate_f(p,i,n, fdict=fdict)*g(p,i,n)*sum_P
    return P_tot
  
    
def raw_S(p,n):
    sum = 0
    for k in range(1,n+1):
        sum += P(p,k,n)*k
    return sum


def calculate_S(p,n, fdict={}, pdict={}):
    sum = 0
    for k in range(1,n+1):
        sum += calculate_P(p,k,n, fdict=fdict, pdict=pdict)*k
    return sum


if __name__ == "__main__":
    # this code is only executed when the script is run rather than imported
    
    
    # READ INPUT ARGUMENTS
    
    # create an argument parser
    parser = argparse.ArgumentParser()
    
    # add all possible arguments that the script accepts
    # and their default values
    parser.add_argument('-p', '--pmin', type=float, default=0.1, 
                        help='Minimum edge probability')
    parser.add_argument('-P', '--pmax', type=float, default=0.6, 
                        help='Maximum edge probability')
    parser.add_argument('-dp', '--dp', type=float, default=0.1, 
                        help='Step size for edge probability')
    parser.add_argument('-n', '--nmin', type=int, default=1, 
                        help='Minimum network size')
    parser.add_argument('-N', '--nmax', type=int, default=500, 
                        help='Maximum network size')
    parser.add_argument('-dn', '--dn', type=int, default=1, 
                        help='Step size for network size')
    parser.add_argument('-ff', '--ffile', type=str, default='fvalues', 
                        help='Path to f file (without file extension)')
    parser.add_argument('-pf', '--pfile', type=str, default='Pvalues', 
                        help='Path to P file (without file extension)')
    parser.add_argument('-ov', '--overwritevalue', type=bool, 
                        default=False, nargs='?', const=True, 
                        help='If True, overwrite existing data values.')
    parser.add_argument('-of', '--overwritefile', type=bool, 
                        default=False, nargs='?', const=True, 
                        help=('If True, do not look for saved data'
                              + ' before writing file. CAREFUL! '
                              + 'THIS MAY REMOVE ALL SAVED DATA!'))
    parser.add_argument('-cf', '--compute-f', type=bool,
                        default=False, nargs='?', const=True,
                        help=('If True, update f data.'))
    parser.add_argument('-cp', '--compute-p', type=bool,
                        default=False, nargs='?', const=True, 
                        help=('If True, update P data.'))   
    
    # parse arguments
    args = parser.parse_args()
    #print(args.__dir__())

    if args.compute_f:
        # LOAD OR MAKE DATA FILES

        # load or make pickle file
        if not args.overwritefile:           

            if os.path.exists(args.ffile+'.p'):    
                # open existing pickle file
                fvalues = pickle.load(open(args.ffile+'.p','rb'))
            else:
                # create an empty dictionary
                fvalues = {}

        else:
            # create an empty dictionary
            fvalues = {}


        # CALCULATE DATA    
        for p in np.arange(args.pmin, args.pmax+args.dp, args.dp):

            t0 = time.time() # take current time

            if p not in fvalues:
                # create a new entry in dictionary if it doesn't exist
                fvalues[p]={}

            for n in range(args.nmin, args.nmax+args.dn, args.dn):

                if n not in fvalues[p]:
                    # create a new entry in dictionary if it doesn't exist
                    fvalues[p][n] = {}

                for i in range(n):

                    # decide if value needs to be computed
                    compute = False

                    if i not in fvalues[p][n]:
                        # compute because data does not exist yet
                        compute = True
                    elif args.overwritevalue:
                        # compute because update requested by user
                        compute = True

                    if compute == True:
                        # calculate f value
                        fval = calculate_f(p,i,n, fdict=fvalues)

                        # add f value to dictionary
                        fvalues[p][n][i] = fval

            # print progress update
            print('f data for p =', "{:.3f}".format(p), 'complete after', 
                  "{:.3f}".format(time.time()-t0), 's')


        # SAVE DATA
        pickle.dump(fvalues, open(args.ffile+'.p','wb'))
        print('Data saved to',args.ffile+'.p')
        
    else:
        # just load existing data for p calculation
        if os.path.exists(args.ffile+'.p'):
            # open existing pickle file
            fvalues = pickle.load(open(args.ffile+'.p','rb'))
        else:
            # create an empty dictionary
            fvalues = {}
                
    if args.compute_p:
        # LOAD OR MAKE DATA FILES

        # load or make pickle file
        if not args.overwritefile:           

            if os.path.exists(args.pfile+'.p'):    
                # open existing pickle file
                pvalues = pickle.load(open(args.pfile+'.p','rb'))
            else:
                # create an empty dictionary
                pvalues = {}

        else:
            # create an empty dictionary
            pvalues = {}


        # CALCULATE DATA    
        for p in np.arange(args.pmin, args.pmax+args.dp, args.dp):

            t0 = time.time() # take current time

            if p not in pvalues:
                # create a new entry in dictionary if it doesn't exist
                pvalues[p]={}

            for n in range(args.nmin, args.nmax+args.dn, args.dn):

                if n not in pvalues[p]:
                    # create a new entry in dictionary if it doesn't exist
                    pvalues[p][n] = {}

                for i in range(n):

                    # decide if value needs to be computed
                    compute = False

                    if i not in pvalues[p][n]:
                        # compute because data does not exist yet
                        compute = True
                    elif args.overwritevalue:
                        # compute because update requested by user
                        compute = True

                    if compute == True:
                        # calculate f value
                        Pval = calculate_P(p,i,n, fdict=fvalues, pdict=pvalues)

                        # add f value to dictionary
                        pvalues[p][n][i] = Pval

            # print progress update
            print('P data for p =', "{:.3f}".format(p), 'complete after', 
                  "{:.3f}".format(time.time()-t0), 's')


        # SAVE DATA
        pickle.dump(pvalues, open(args.pfile+'.p','wb'))
        print('Data saved to',args.pfile+'.p')
       
                
                
            
            
  
  
  