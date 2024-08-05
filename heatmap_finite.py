import sys, pickle, time, os
sys.path.insert(0, "C:\\Users\\f00689q\\My Drive\\jupyter\\small-perc\\libs")

import numpy as np

from scipy.special import comb
from scipy.integrate import simpson
from scipy.signal import argrelextrema
from random import choice

from libs.utils import *
from libs.finiteTheory import *
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
attack = True

path = os.path.join('C:\\Users\\f00689q\\My Drive\\jupyter\\small-perc\\data', 'heatmaps', 'p{:.2f}'.format(p))
if not os.path.exists(path):
    os.mkdir(path)

for i in range(0,100,1):

    t0 = time.time()
    n = i+1
    name = 'relSCurve_attack{}_n{}_p{:.2f}'.format(attack,n,p)

    print ('Number of nodes:', n)

    fin_curve = relSCurve(p, n,
        attack=attack, fdict=fvals, pdict=pvals,
        lcc_method_relS="pmult", executable_path="C:\\Users\\f00689q\\My Drive\\jupyter\\small-perc\\libs\\p-recursion.exe")

    np.save(os.path.join(path,'{}.npy'.format(name)), fin_curve)

    print (os.path.join(path,'{}.npy'.format(name)), 'saved after', time.time()-t0)