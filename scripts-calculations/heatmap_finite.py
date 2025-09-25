# conduct a 2d parameter sweep for n and p of the expected LCC size under node removal
# expected LCC size is computed using the finite theory
# Input: float p (edge probability)
# Input files: fvalues.p, Pvalues.p under `repository-root/data-synthetic/` (optional)
# Output files: Finite theory results stored in 100 files (each corresponding to a 
# different value of the network size parameter n) under 
# `repository-root/data/synthetic/` as `
# relSCurve{num_trials}_attack{bool}_n{n}_p{p:.2f}.npy`

# Import libraries
import sys, time, os
import numpy as np
from pathlib import Path

# Add the parent directory to the path to import local libraries
REPO_ROOT = str(Path(__file__).parent.parent)
CACHE_PATH = os.path.join(REPO_ROOT, 'cache-combinatorics')
SYNTH_PATH = os.path.join(REPO_ROOT, 'data-synthetic')
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Import from local libraries
from libs.utils import load_recursion_cache
from libs.finiteTheory import relative_lcc_sequence

# use cached values or recalculate (toggle as needed)
#fvals, pvals = {}, {} 
fvals, pvals = load_recursion_cache(CACHE_PATH)

# get p from command line
p = float(sys.argv[1])
attack = True # toggle as needed

# Start calculating data
for i in range(0,100,1):

    t0 = time.time()
    n = i+1
    fname = 'relSCurve_attack{}_n{}_p{:.2f}'.format(attack,n,p)

    print ('Number of nodes:', n)

    fin_curve = relative_lcc_sequence(p, n, targeted_removal=attack, 
        connectivity_cache=fvals, probability_cache=pvals,
        method="external", executable_name='p-recursion.exe')

    fpath = os.path.join(SYNTH_PATH, f'{fname}.npy')
    np.save(fpath, fin_curve)
    print (fpath, 'saved after', time.time()-t0)