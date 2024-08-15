import sys, pickle, time, os
sys.path.insert(0, "C:\\Users\\f00689q\\My Drive\\jupyter\\small-perc\\libs")

import numpy as np

from scipy.special import comb
from scipy.integrate import simpson
from scipy.signal import argrelextrema
from random import choice

from libs.utils import *
# from libs.finiteTheory import *
#from visualizations import *
#from libs.utils import *
from libs.robustnessSimulations import *
#from performanceMeasures import *
#from infiniteTheory import *
#from finiteTheory import *

fvals = {} #pickle.load(open('data/fvalues.p', 'rb'))
pvals = {} #pickle.load(open('data/Pvalues.p', 'rb'))

# get p from command line
p = float(sys.argv[1])
attack = True

if attack:
    remove_strategies = ['attack']
else:
    remove_strategies = ['random']

path = os.path.join('C:\\Users\\f00689q\\My Drive\\jupyter\\small-perc\\data', 'synthetic_data', 'p{:.2f}'.format(p))
if not os.path.exists(path):
    os.mkdir(path)

for i in range(0,100,1):

    t0 = time.time()
    n = i+1
    name = 'simRelSCurve_attack{}_n{}_p{:.2f}'.format(attack,n,p)

    print ('Number of nodes:', n)
    data = completeRCData(numbers_of_nodes=[n], edge_probabilities=[p],
        num_trials=100, performance='relative LCC',
        graph_types=['ER'], remove_strategies=remove_strategies)[0][0][0][0][1:]
    #data = 
    print('data', np.array(data).shape)

    np.save(os.path.join(path,'{}.npy'.format(name)), data)

    print (os.path.join(path,'{}.npy'.format(name)), 'saved after', time.time()-t0)
    