# conduct a 2d parameter sweep for n and p of the expected LCC size under node removal
# expected LCC size is computed using the infinite theory
# conduct a 2d parameter sweep for n and p of the expected LCC size under node removal
# expected LCC size is computed using the finite theory
# Input: float p (edge probability)
# Output files: Finite theory results stored in 100 files (each corresponding to a 
# different value of the network size parameter n) under 
# `repository-root/data/synthetic/` as `
# infRelSCurve{num_trials}_attack{bool}_n{n}_p{p:.2f}.npy`

# Import libraries
import sys, time, os
import numpy as np
from scipy.special import comb
from scipy.integrate import simpson
from scipy.signal import argrelextrema
from random import choice
from pathlib import Path

# Add the parent directory to the path to import local libraries
REPO_ROOT = str(Path(__file__).parent.parent)
SYNTH_DATA = os.path.join(REPO_ROOT, 'data-synthetic')
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Import from local libraries
from libs.utils import *
from libs.infiniteTheory import relative_lcc_sequence

# get p from command line
p = float(sys.argv[1])

for attack in [False, True]:

    for i in range(0,100,1):

        t0 = time.time()
        n = i+1
        file_name = 'infRelSCurve_attack{}_n{}_p{:.2f}.npy'.format(attack,n,p)

        print ('Number of nodes:', n)

        infin_curve = relative_lcc_sequence(n, p, targeted_removal=attack)

        file_path = os.path.join(SYNTH_DATA,file_name)

        np.save(file_path, infin_curve)

        print (file_name, 'saved after', time.time()-t0)