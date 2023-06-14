# script that executes the plotting function

# import packages
from matplotlib import pyplot as plt
import sys, pickle
# include the directory with libraries in the pythonpath
sys.path.insert(0, "libs")

# import local libraries
from visualizations import *

# start script
if __name__ == "__main__":

    # decide what to do with the figure
    save = False
    show = False

    # import data
    fvals = pickle.load(open('data/fvalues.p', 'rb'))
    pvals = pickle.load(open('data/Pvalues.p', 'rb'))

    # make figure
    fig = plot_graphs(numbers_of_nodes=[20], edge_probabilities=[0.1],
        graph_types=['ER'], remove_strategies=["random"], num_trials=10,
        performance='relative LCC', smooth_end=True,
        forbidden_values=[0], fdict=fvals, pdict=pvals, 
        savefig='figures/x.png')

    if save:
        fig.savefig('figures/test.png')

    if show:
        fig.show()