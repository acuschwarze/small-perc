###############################################################################
#
# This is a script to generate data for the recursion equation used in the
# finite theory for percolation on small networks.
#
# The script has the following command line arguments:
#     --pmin (-p): Minimum edge probability (default=0.1)
#     --pmin (-P): Maximum edge probability (default=0.6)
#     --dp(-dp): Step size for edge probability (default=0.1)
#     --nmin(-n): Minimum network size (default=1)
#     --nmax(-N): Maximum network size (default=500)
#     --dn(-dn): Step size for network size (default=1)
#     --ffile (-ff):  Path to f file without file extension 
#        (default='data/fvalues')
#     --pfile (-pf):  Path to P file without file extension
#        (default='data/Pvalues')
#     --overwritevalue (-ov): If True, overwrite existing data values.
#        (default=False)
#     --overwritevalue (-ov): f True, do not look for saved data before writing
#        file. CAREFUL! THIS MAY REMOVE ALL SAVED DATA! (default=False)
#     --compute-f (-cf): If True, update existing f data. (default=False)
#     --compute-p (-cp): If True, update existing p data. (default=False)
#
# Default settings assume that there is a "data" folder in the same directory
# as this file.
#
###############################################################################

# IMPORT PACKAGES
import numpy as np
from scipy.special import comb
from random import choice
import sys, argparse # import packages to manage input arguments for scripts
import os, pickle, csv # import packages for file I/O
import time # package to help keep track of calculation time
from libs.finiteTheory import *

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
    parser.add_argument('-ff', '--ffile', type=str, default='data/fvalues',
                        help='Path to f file (without file extension)')
    parser.add_argument('-pf', '--pfile', type=str, default='data/Pvalues',
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
                        help=('If True, update existing f data.'))
    parser.add_argument('-cp', '--compute-p', type=bool,
                        default=False, nargs='?', const=True,
                        help=('If True, update existing P data.'))
    
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
       
                
                
            
            
  
  
  