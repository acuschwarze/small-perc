# conduct a 2d parameter sweep for n and p of the expected LCC size under node removal
# expected LCC size is computed using the infinite theory

import sys, pickle, time, os
sys.path.insert(0, "C:\\Users\\f00689q\\My Drive\\jupyter\\small-perc\\libs")

import numpy as np

from scipy.special import comb
from scipy.integrate import simpson
from scipy.signal import argrelextrema
from random import choice

from libs.utils import *
from libs.infiniteTheory import *
#from visualizations import *
#from libs.utils import *
#from robustnessSimulations import *
#from performanceMeasures import *
#from infiniteTheory import *
#from finiteTheory import *

fvals = {} #pickle.load(open('data/fvalues.p', 'rb'))
pvals = {} #pickle.load(open('data/Pvalues.p', 'rb'))

# get p from command line
p = float(sys.argv[1])
attack = False

path = os.path.join('C:\\Users\\f00689q\\My Drive\\jupyter\\small-perc\\data', 'synthetic_data', 'p{:.2f}'.format(p))
if not os.path.exists(path):
    os.mkdir(path)

for i in range(0,100,1):

    t0 = time.time()
    n = i+1
    name = 'infRelSCurve_attack{}_n{}_p{:.2f}'.format(attack,n,p)

    print ('Number of nodes:', n)

    infin_curve = relSCurve(n, p, attack=attack)

    np.save(os.path.join(path,'{}.npy'.format(name)), infin_curve)

    print (os.path.join(path,'{}.npy'.format(name)), 'saved after', time.time()-t0)