import random
import numpy as np
from scipy.special import comb as C
import matplotlib.pyplot as plt
import pandas as pd
import igraph as ig

def how_many_removal_options(n, m, k):
    each_nodes_edges_at_start = n - 1
    edges_to_remove = C(n, 2, exact=True) - m
    safe_to_remove_from_any = min(n - 1 - k, edges_to_remove)

    initial_removal_options = C(n, safe_to_remove_from_any, exact=True)

    edges_left_to_remove = edges_to_remove - safe_to_remove_from_any

    # This next part is tricky...
    secondary_removal_options = []
    for were_removed_from_key in range(0, safe_to_remove_from_any + 1):
        were_removed_from_other = safe_to_remove_from_any - were_removed_from_key

        probability = C(n - 1, were_removed_from_key, exact=True) / initial_removal_options
        options = C(C(n - 1, 2, exact=True) - were_removed_from_other, edges_left_to_remove, exact=True)

        secondary_removal_options.append(probability * options)
    return initial_removal_options * np.sum(secondary_removal_options)


def probability_of_atleast_degree(n, m, k):
    ways = how_many_removal_options(n, m, k)
    all_combos = C(C(n, 2, exact=True), m, exact=True)

    for_single = ways / all_combos

    if for_single > 1 / n:
        return 1.0
    else:
        return n * for_single


def maxdeg_estimate(n, m):
    estimate = 0

    highest_k = n - 1
    higher_than = 0

    for k in range(highest_k, -1, -1):
        probability = probability_of_atleast_degree(n, m, k) * (1 - higher_than)
        estimate += probability * k

        higher_than += probability
    return estimate


def p_atmost_k(n, p, k):
    prob = 0
    for k_ in range(0, k + 1):
        prob += C(n - 1, k_, exact=True) * p ** k_ * (1 - p) ** (n - 1 - k_)
    return prob


def p_maxdeg_estimate(n, m):
    p = m / C(n, 2, exact=True)

    estimate = 0

    highest_k = n - 1
    lower_than = 0

    for k in range(0, highest_k + 1):
        probability = (p_atmost_k(n, p, k) ** n) - lower_than
        lower_than += probability

        estimate += probability * k

    return estimate



n = 10
iterations = 20

edges = [[(i, j) for i in range(j + 1, n)] for j in range(n)]
edges = [item for sublist in edges for item in sublist]

G = ig.Graph(n, edges)
DF = pd.DataFrame(index=range(len(edges) + 1), columns=['m-predicted', 'p-predicted', 'actual'])

actual = {}
for i in range(len(edges) + 1):
    actual[i] = []
for _ in range(iterations):
    G = ig.Graph(n, edges)

    e_num = len(G.es)
    while e_num > 0:
        actual[e_num].append(max(G.degree()))
        G.delete_edges(random.choice(G.es))

        e_num = len(G.es)
    actual[e_num].append(max(G.degree()))

for e_num in range(len(edges) + 1):
    DF['m-predicted'][e_num] = maxdeg_estimate(n, e_num)
    DF['p-predicted'][e_num] = p_maxdeg_estimate(n, e_num)
    DF['actual'][e_num] = np.mean(actual[e_num])

for itr in range(iterations):
    plt.plot(range(len(DF)), [actual[dt][itr] for dt in actual], color='lightblue', alpha=0.5, linestyle='dotted',
             linewidth=0.4)

plt.plot(range(len(DF)), DF['actual'])
plt.plot(range(len(DF)), DF['m-predicted'])
plt.plot(range(len(DF)), DF['p-predicted'])

plt.title("Comparison of m-Predicted (orange) and p-Predicted(green) to Actual (blue) maximum degrees; n=" + str(n))

plt.xlabel("# of edges")
plt.ylabel("max. degree")

plt.show()