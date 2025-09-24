# Generate synthetic percolation data for random networks
# Input: float p (edge probability)
# Output files: Simulation data stored in 100 files (each corresponding to a 
# different value of the network size parameter n) under 
# `repository-root/data/synthetic/` as `
# simRelSCurve{num_trials}_attack{bool}_n{n}_p{p:.2f}.npy`

# Import libraries
import sys, time, os
import numpy as np
from pathlib import Path

# Add the parent directory to the path to import local libraries
REPO_ROOT = str(Path(__file__).parent.parent)
SYNTH_PATH = os.path.join(REPO_ROOT, 'data-synthetic')
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Import from local libraries
from libs.robustnessSimulations import robustness_sweep


# get p from command line
p = float(sys.argv[1])

# set other parameters
attack = True # toggle as needed
num_trials = 1000

# start data generation
remove_strategies = [('attack' if attack else 'random')]
for i in range(0,100,1):

    t0 = time.time()
    n = i+1
    fname = 'simRelSCurve{}_attack{}_n{}_p{:.2f}'.format(num_trials,attack,n,p)

    print ('Compute data for number of nodes:', n)
    data = robustness_sweep(numbers_of_nodes=[n], edge_probabilities=[p],
        num_trials=num_trials, performance='relative LCC',
        graph_types=['ER'], remove_strategies=remove_strategies)[0][0][0][0][1:]
    
    fpath = os.path.join(SYNTH_PATH, f'{fname}.npy')
    np.save(fpath, data)
    print (fpath, 'saved after', time.time()-t0)
    