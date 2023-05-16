import numpy as np
import scipy
from scipy.special import comb
import scipy.stats as sst
import networkx as nx
from random import choice
import matplotlib.pyplot as plt
from data import *

def f(p,i,n): #original recursive function for f
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

def raw_f(p, i, n): # I think this is the same as above
    if i == 0:
        p_connect = 0
    if i == 1:
        p_connect = 1
    else:
        sum_f = 0
        for i_n in range(1, i, 1):
            sum_f += f(p, i_n, n) * scipy.special.comb(i - 1, i_n - 1) * (1 - p) ** ((i_n) * (i - i_n))
        p_connect = 1 - sum_f
    return p_connect


def calculate_f(p, i, n, fdict={}): # using dictionary to calculate f values
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
        for i_n in range(1, i, 1):
            sum_f += calculate_f(p, i_n, n, fdict=fdict) * scipy.special.comb(i - 1, i_n - 1) * (1 - p) ** (
                        (i_n) * (i - i_n))
        p_connect = 1 - sum_f
    #print("f")
    #print(p_connect)
    return p_connect


def g(p, i, n): # function to calculate g values
    return (1 - p) ** (i * (n - i))

def P(p,i,n): # original P value function
  #print("execute P", p, i, n)
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

def raw_P(p, i, n): # also same as original P value function
    if i == 0 and n == 0:
        P_tot = 1
    elif i > 0 and n == 0:
        P_tot = 0
    elif i > n or n < 0 or i <= 0:
        P_tot = 0
    elif i == 1 and n == 1:
        P_tot = 1
    elif i == 1 and n != 1:
        P_tot = (1 - p) ** scipy.special.comb(n, 2)
    else:
        sum_P = 0
        for j in range(0, i + 1, 1):  # shouldn't it be i+1?
            sum_P += P(p, j, n - i)
        P_tot = scipy.special.comb(n, i) * f(p, i, n) * g(p, i, n) * sum_P
    return P_tot


def calculate_P(p, i, n, fdict={}, pdict={}): #using dictionaries to calculate P
    if p in pdict:
        if n in pdict[p]:
            if i in pdict[p][n]:
                return pdict[p][n][i]

    if i == 0 and n == 0:
        P_tot = 1
    elif i > 0 and n == 0:
        P_tot = 0
    elif i > n or n < 0 or i <= 0:
        P_tot = 0
    elif i == 1 and n == 1:
        P_tot = 1
    elif i == 1 and n != 1:
        P_tot = (1 - p) ** scipy.special.comb(n, 2)
    else:
        sum_P = 0
        for j in range(0, i + 1, 1):  # shouldn't it be i+1?
            sum_P += calculate_P(p, j, n - i, fdict=fdict, pdict=pdict)
        P_tot = scipy.special.comb(n, i) * calculate_f(p, i, n, fdict=fdict) * g(p, i, n) * sum_P
    #print("P")
    #print(P_tot)
    return P_tot


def raw_S(p, n): #original way to calculate S
    sum = 0
    for k in range(1, n + 1):
        sum += P(p, k, n) * k
    return sum


def calculate_S(p, n, fdict={}, pdict={}): # calculating S with dictionaries
    sum = 0
    for k in range(1, n + 1):
        sum += calculate_P(p, k, n, fdict=fdict, pdict=pdict) * k
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
    # print(args.__dir__())

    if args.compute_f:
        # LOAD OR MAKE DATA FILES

        # load or make pickle file
        if not args.overwritefile:

            if os.path.exists('fvalues.p' + '.p'):
                # open existing pickle file
                fvalues = pickle.load(open(args.ffile + '.p', 'rb'))
            else:
                # create an empty dictionary
                fvalues = {}

        else:
            # create an empty dictionary
            fvalues = {}

        # CALCULATE DATA
        for p in np.arange(args.pmin, args.pmax + args.dp, args.dp):

            t0 = time.time()  # take current time

            if p not in fvalues:
                # create a new entry in dictionary if it doesn't exist
                fvalues[p] = {}

            for n in range(args.nmin, args.nmax + args.dn, args.dn):

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
                        fval = calculate_f(p, i, n, fdict=fvalues)

                        # add f value to dictionary
                        fvalues[p][n][i] = fval

            # print progress update
            print('f data for p =', "{:.3f}".format(p), 'complete after',
                  "{:.3f}".format(time.time() - t0), 's')

        # SAVE DATA
        pickle.dump(fvalues, open(args.ffile + '.p', 'wb'))
        print('Data saved to', args.ffile + '.p')

    else:
        # just load existing data for p calculation
        if os.path.exists(args.ffile + '.p'):
            # open existing pickle file
            fvalues = pickle.load(open(args.ffile + '.p', 'rb'))
        else:
            # create an empty dictionary
            fvalues = {}

    if args.compute_p:
        # LOAD OR MAKE DATA FILES

        # load or make pickle file
        if not args.overwritefile:

            if os.path.exists(args.pfile + '.p'):
                # open existing pickle file
                pvalues = pickle.load(open(args.pfile + '.p', 'rb'))
            else:
                # create an empty dictionary
                pvalues = {}

        else:
            # create an empty dictionary
            pvalues = {}

        # CALCULATE DATA
        for p in np.arange(args.pmin, args.pmax + args.dp, args.dp):

            t0 = time.time()  # take current time

            if p not in pvalues:
                # create a new entry in dictionary if it doesn't exist
                pvalues[p] = {}

            for n in range(args.nmin, args.nmax + args.dn, args.dn):

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
                        Pval = calculate_P(p, i, n, fdict=fvalues, pdict=pvalues)

                        # add f value to dictionary
                        pvalues[p][n][i] = Pval

            # print progress update
            print('P data for p =', "{:.3f}".format(p), 'complete after',
                  "{:.3f}".format(time.time() - t0), 's')

        # SAVE DATA
        pickle.dump(pvalues, open(args.pfile + '.p', 'wb'))
        print('Data saved to', args.pfile + '.p')


#get f and p values
fvalues = pickle.load(open('fvalues.p', 'rb'))
pvalues = pickle.load(open('Pvalues.p', 'rb'))


def big_S(p, n, fdict1=fvalues, pdict1=pvalues): #usually i input an array with one n value for now
    S_array = np.zeros(n[0]) #takes that n value and makes a new array with that length
    for i in range(1,n[0]+1): #indexing purposes
        S_array[i-1] = calculate_S(p, i, fdict=fdict1, pdict=pdict1) #finds S for each value <= n
    return S_array

def big_relS(p, n, fdict1=fvalues, pdict1=pvalues): #i also input just one n value
    S_array = np.zeros(n[0])
    for i in range(1,n[0]+1):
        S_array[i-1] = calculate_S(p, i, fdict=fdict1, pdict=pdict1)/i # divide by i to get the rel LCC
    return S_array


def S_calc_data(p=.1,n=[20,50,100]): #returns the values of S for multiple n values.
  x_array = np.zeros(len(n))
  y_array = np.zeros(len(n))
  for a in range(len(n)): # i think this is actually the same as n?
    x_array[a] = n[a]
  for b in range(len(n)):
    y_array[b] = calculate_S(p,n[b])
  print(x_array, y_array)
  return x_array, y_array


# start of robustness library stuff

# performance measures
def average_degree(graph): # returns mean degree
    # print("mean degree" + str(graph.number_of_edges() * 2 / graph.number_of_nodes()))
    return graph.number_of_edges() * 2 / graph.number_of_nodes()


def compute_efficiency(G):
    # average shortest path between all pairs of nodes
    '''This function takes a networkx graph and computes its efficiency.'''
    # print(apsp)
    n = G.number_of_nodes()
    if n < 2:
        return 0  # networks with 1 or 0 nodes can easily reach "all" their neighbors
    apsp = list(nx.all_pairs_shortest_path(G))
    pairwise_efficiencies = np.zeros((n, n))
    for i_index, i in enumerate(G.nodes()):
        # print('i', i)
        for tup in apsp:
            if tup[0] == i:
                path_dict = tup[1]
        # path_dict = apsp[i_index][1]
        for j_index, j in enumerate(G.nodes()):
            # print('j',j)
            if i != j:
                if j in path_dict.keys():
                    p = path_dict[j]
                    if len(p) > 1:
                        pairwise_efficiencies[i_index, j_index] = 1 / (len(p) - 1)
    efficiency = np.sum(pairwise_efficiencies) / (n * (n - 1))
    return efficiency


def degree_distr(x, G):
    # returns fraction of nodes in graph G with x edges
    array = G.degree
    degree_count = 0
    for node in G.degree:
        if node[1] == x:
            degree_count += 1
    return degree_count / G.number_of_nodes()


def mean_shortest_path(graph):
    if nx.number_of_nodes(graph) == 0:
        return 0
    else:
        n = len(max(nx.connected_components(graph), key=len))
        if n < 2:
            return np.nan
        apsp = list(nx.all_pairs_shortest_path(graph))
        shortest_distances = np.zeros((n, n))
        for i_index, i in enumerate(max(nx.connected_components(graph), key=len)):
            # print('i', i)
            for tup in apsp:
                if tup[0] == i:
                    path_dict = tup[1]
            # path_dict = apsp[i_index][1]
            for j_index, j in enumerate(max(nx.connected_components(graph), key=len)):
                # print('j',j)
                if i != j:
                    if j in path_dict.keys():
                        p = path_dict[j]
                        if len(p) > 1:
                            shortest_distances[i_index, j_index] = len(p) - 1
        mean_shortest = np.sum(shortest_distances) / scipy.special.comb(n, 2)
        return mean_shortest


def communicability(G):
    n = G.number_of_nodes()
    if n < 2:
        return 0  # networks with 1 or 0 nodes can easily reach "all" their neighbors

    big_comm_dict = nx.communicability(G)
    comm_matrix = np.zeros((n, n))
    for i_index, i in enumerate(G.nodes()):
        # print('i', i)
        for start in big_comm_dict:
            if start == i:
                small_comm_dict = big_comm_dict[start]
            # comm_dict = comm_list[i_index][1]
        for j_index, j in enumerate(G.nodes()):
            # print('j',j)
            # if i != j:
            if j in small_comm_dict.keys():
                c = small_comm_dict[j]
                comm_matrix[i_index, j_index] = c
    return np.log(np.trace(comm_matrix)) - np.log(n)


def laplacian_matrix(G):
    n = G.number_of_nodes()
    if n < 1:
        return "none"  # networks with 1 or 0 nodes can easily reach "all" their neighbors
    apsp = list(nx.all_pairs_shortest_path(G))
    L = np.zeros((n, n))
    for i_index, i in enumerate(G.nodes()):
        for tup in apsp:
            if tup[0] == i:
                path_dict = tup[1]
        # path_dict = apsp[i_index][1]
        for j_index, j in enumerate(G.nodes()):
            if i == j:
                L[i_index, j_index] = G.degree(i)
            if i != j:
                if j in path_dict.keys():
                    p = path_dict[j]
                    if len(p) == 2:
                        L[i_index, j_index] = -1
                    else:
                        L[i_index, j_index] = 0
    return L


def computeRobustnessCurve(g, remove_nodes='random', performance='largest_connected_component'):
    # takes in a graph, removal strategy, and performance metric to measure robustness over one trial
    # helper function for computeRobustnessCurves

    # compute the maximum number of nodes to be removed
    n = g.number_of_nodes()

    data_array = np.zeros((2, n), dtype=float)
    # 2 rows of n columns each: top row is number of nodes removed, bottom row is performance measurement
    data_array[0] = np.arange(n)

    if performance == 'number_of_nodes':

        def computePerformance(graph):
            output = graph.number_of_nodes()
            return output

    elif performance == 'largest_connected_component':
        # number of nodes in largest connected component of a graph

        def computePerformance(graph):
            if (graph.number_of_nodes() == 0):
                return 0
            else:
                nodes_in_cluster = len(max(nx.connected_components(graph), key=len))
                return nodes_in_cluster

    elif performance == 'efficiency':

        def computePerformance(graph):
            return compute_efficiency(graph)

    elif performance == "entropy":
        # measure of how "spread out/heterogeneous" the degree distribution of a graph is

        def computePerformance(graph):
            if nx.number_of_nodes(graph) == 0:
                return 0
            else:
                max_degree = sorted(graph.degree, key=lambda x: x[1], reverse=True)[0][1]
                H = 0
                for i in range(max_degree):
                    if degree_distr(i, graph) == 0:
                        p_k = 0
                    else:
                        p_k = degree_distr(i, graph)
                        H += p_k * np.log(p_k)
                    # p_k = degree_distr(i, graph)
                    # H += p_k * np.log(p_k)
                return -H

    elif performance == "mean shortest path":
        def computePerformance(graph):
            return mean_shortest_path(graph)

    elif performance == "average cluster size":
        def computePerformance(graph):
            n = nx.number_of_nodes(graph)
            n_c = nx.number_connected_components(graph)
            if n_c == 0:
                return 0
            return n / n_c

    elif performance == "average small component size":
        def computePerformance(graph):
            n = nx.number_of_nodes(graph)
            if n == 0:
                lcc = 0
            else:
                lcc = len(max(nx.connected_components(graph), key=len))
            n_no_lcc = n - lcc
            n_c = nx.number_connected_components(graph) - 1
            if n_c == 0:
                return 0
            return n_no_lcc / n_c

    elif performance == "average small component size smooth":
        def computePerformance(graph):
            n = nx.number_of_nodes(graph)
            k = average_degree(graph)
            p_t = 1 / k
            if n == 0:
                lcc = 0
            else:
                lcc = len(max(nx.connected_components(graph), key=len))
            n_no_lcc = n - lcc
            n_c = nx.number_connected_components(graph) - 1
            if n_c == 0:
                return 0
            i
            return n_no_lcc / n_c

    elif performance == "relative LCC":

        def computePerformance(graph):
            if (graph.number_of_nodes() == 0):
                return 0
            else:
                nodes_in_cluster = len(max(nx.connected_components(graph), key=len))
                return nodes_in_cluster / graph.number_of_nodes()

    elif performance == "reachability":
        # proportion of pairs of nodes between which there exists a path
        def computePerformance(graph):
            N = graph.number_of_nodes()
            if N == 0:
                return 0
            else:
                reach = 0
                # replace
                for i_index, i in enumerate(graph.nodes()):
                    for j_index, j in enumerate(graph.nodes()):
                        if i != j:
                            if nx.has_path(graph, i, j) == True:
                                reach += 1

            return .5 * 1 / (scipy.special.comb(N, 2)) * reach

    elif performance == "transitivity":
        # interconnectedness of adjacent nodes; measures incidence of clusters/cliques
        def computePerformance(graph):
            return nx.transitivity(graph)

    elif performance == "resistance distance":
        def computePerformance(graph):
            N = graph.number_of_nodes()
            if N < 1:
                return 0
            L = laplacian_matrix(graph)
            L_plus = np.linalg.pinv(L)

            return N * np.trace(L_plus)

    elif performance == "natural connectivity":
        def computePerformance(graph):
            return communicability(graph)

    else:
        print('Error: I dont know that performance value')
        return 0

    for i in range(n):
        # find a node to remove
        if remove_nodes == 'random':
            v = choice(list(g.nodes()))
        elif remove_nodes == 'attack':
            v = sorted(g.degree, key=lambda x: x[1], reverse=True)[0][0] #find most connected node
        else:
            print('Error: I dont know that mode of removing nodes')
            v = None  # will this error
        g.remove_node(v)

        data_array[1, i] = computePerformance(g) # calculates performance value after each removal

        if performance == "average small component size:":
            # this was after we started doing stuff from the book - this helps calculate little s
            n = nx.number_of_nodes(g)
            n_c = nx.number_connected_components(g)
            k = average_degree(g)
            p_t = 1 / k #percolation threshold
            if i in np.arange(0, (1 / k) * g.number_of_nodes(), 1):
                data_array[1, i] = n / n_c

    return data_array


# constructs an ER(random) or SF (exponential) network–helper function for computeRobustnessCurves and computeRobustnessData

def construct_a_network(number_of_nodes, number_of_edges, graph_type):  # n = #nodes, m = #edges
    # density = .1   #used this when we wanted a specific density rather than a specific m
    # number_of_edges = int((number_of_nodes - 1) * density)/2  #used this to calculate m from density
    if graph_type == 'ER':
        p=.1 #setting a specific p for the recent stuff with lambert/recursive rel LCC
        #p = 2 * number_of_edges * (number_of_nodes - number_of_edges) / (number_of_nodes * (number_of_nodes - 1))
        g = nx.erdos_renyi_graph(number_of_nodes, p, seed=None, directed=False)
        #checking that things are calculated correctly
        # print("ER" + str(number_of_nodes) + "mean degree" + str(average_degree(g)) + "density" + str(nx.density(g)))
        return g
    elif graph_type == 'SF':
        g2 = nx.barabasi_albert_graph(number_of_nodes, number_of_edges)
        #checking that things are running correctly
        # print("SF" + str(number_of_nodes) + "mean degree" + str(average_degree(g2)) + "density" + str(nx.density(g2)))
        return g2
    else:
        print("Error: invalid graph_type")


def computeRobustnessCurves(number_of_nodes=100, number_of_edges=20, graph_type='ER', remove_nodes='random',
                            performance='largest_connected_component', num_trials=10):
    # constructs a network and runs computeRobustnessCurves over given amount of trials; helper for completeRobustnessData

    # array to store data from multiple trials
    data_array = np.zeros((num_trials + 1, number_of_nodes), dtype=float)
    data_array[0] = np.arange(number_of_nodes)

    for i in range(num_trials):
        g = construct_a_network(number_of_nodes=number_of_nodes, number_of_edges=number_of_edges, graph_type=graph_type)
        if average_degree(g) == 0:
            percolation_threshold = 0
        else:
            #not sure why both are here? Maybe to calculate percolation with two different methods?
            percolation_threshold = 1 / average_degree(g)
            percolation_threshold = 1 / number_of_nodes + (number_of_nodes - 1) / (average_degree(g) * number_of_nodes)
            # print(percolation_threshold)

        data = computeRobustnessCurve(g, remove_nodes=remove_nodes, performance=performance)
        data_array[i + 1] = data[1] # add a new row of performance data

    return data_array, percolation_threshold  # 2d array for performance over removing n nodes, num_trials trials


def perf_sim2(n, p, smoothing=False):
    # y_array = np.zeros(n-2)
    # x_array = np.zeros(n-2)
    # for i in range(n-2):
    y_array = np.zeros(n)
    x_array = np.zeros(n)
    for i in range(n):
        x_array[i] = i / n #get proportion of nodes "removed"

    new_p = p # could probably delete this now since p doesn't change
    new_n = n

    for i in range(n):
        c = 2 * new_p * scipy.special.comb(new_n, 2) / new_n #mean degree
        if smoothing == True:
            y_array[i] = 1 + scipy.special.lambertw((-c * np.exp(-c)), k=0, tol=1e-8) / c + 1 / new_n
        else:
            y_array[i] = 1 + scipy.special.lambertw((-c * np.exp(-c)), k=0, tol=1e-8) / c
        if new_n ** 2 - 2 * new_n == 0: # can delete this if else statement
            new_p = new_p
        else:
            new_p = new_p
            new_n -= 1

    return x_array, y_array


def my_lambertw(x, k=0): # lambert function with the values near percolation threshold guaranteed (sometimes
                        # just doing lambert function will leave small gaps in the graph)
    if x + 1 / np.exp(1) < 1E-20:
        return -1.0
    else:
        return scipy.special.lambertw(x, k=k)


def perf_sim2copy(n, p, smoothing=False): #this was a function for troubleshooting lambert stuff
    # y_array = np.zeros(n-2)
    # x_array = np.zeros(n-2)
    # for i in range(n-2):
    # we did -2 to avoid the end tail bit at first
    y_array = np.zeros(n - 2) # just the part that goes into the lambert function
    z_array = np.zeros(n - 2) # actual S from entire lambert calculation
    x_array = np.zeros(n - 2) # proportion of nodes "removed"
    mean_array = np.zeros(n) #for c values
    difference = np.zeros(n) #for difference between y and 1/e - made sure some values aligned, but I can't quite
                            #remember why - I don't think we use this much anymore though

    new_p = p
    new_n = n
    c_init = 2 * new_p * scipy.special.comb(new_n, 2) / new_n
    percolation_threshold2 = 1 / n + (n - 1) / (c_init * n)

    for i in range(n - 2):
        x_array[i] = i / n

    for i in range(n - 2):
        c = 2 * new_p * scipy.special.comb(new_n, 2) / new_n
        if smoothing == True:
            y_array[i] = -c * np.exp(-c)
            z_array[i] = 1 + my_lambertw(-c * np.exp(-c)) / c + 1 / new_n
        else:
            y_array[i] = -c * np.exp(-c)
            z_array[i] = 1 + my_lambertw(-c * np.exp(-c)) / c
            mean_array[i] = c
            difference[i] = y_array[i] - 1 / scipy.e
        if new_n ** 2 - 2 * new_n == 0:
            new_p = new_p
        else:
            new_p = new_p
            new_n -= 1

    return x_array, y_array, z_array, mean_array, difference, percolation_threshold2


def maxdegree(n, p): #for the attack simluations with lambert function etc
    vals = [[i, n * sst.binom(n, p).pmf(i)] for i in range(1, n + 1)]
    for i, v in vals[::-1]:
        if v >= 1:
            return i
    return 0


def perf_sim2_attack(n, p, smoothing=False): #does the lambert calculations but with attacks
    y_array = np.zeros(n)
    x_array = np.zeros(n)
    for i in range(n):
        x_array[i] = i / n

    new_p = p #this time probably can't remove this
    new_n = n

    for i in range(n - 2):
        max = maxdegree(new_n, new_p)
        c = 2 * new_p * scipy.special.comb(new_n, 2) / new_n
        if smoothing == True:
            y_array[i] = 1 + scipy.special.lambertw((-c * np.exp(-c)), k=0, tol=1e-8) / c + 1 / new_n
        else:
            y_array[i] = 1 + scipy.special.lambertw((-c * np.exp(-c)), k=0, tol=1e-8) / c

        if new_n ** 2 - 2 * new_n == 0:
            new_p = new_p
        else:
            new_p = new_p * new_n / (new_n - 2) - 2 * max / ((new_n - 1) * (new_n - 2))
            new_n -= 1

    return x_array, y_array


def perf_sim_revision1(n, p, smoothing=False): #I think this ended up not working well. Could delete
    y_array = np.zeros(n)
    x_array = np.zeros(n)
    for i in range(n):
        x_array[i] = i / n

    new_p = p
    new_n = n
    for i in range(n - 2):
        c = 2 * new_p * scipy.special.comb(new_n, 2) / new_n

        # the function below is from the book - before they approximate using n to infinity
        func = lambda x: (new_n - 1) * np.log(1 - c / (new_n - 1) * (1 - x)) - np.log(x)
        # I don't think this function actually does anything with S
        S = 1 + scipy.special.lambertw((-c * np.exp(-c)), k=0, tol=1e-8) / c
        # solving the function for the rel lcc - but because of logs, there were lots of random jumps
        S2 = scipy.optimize.fsolve(func, np.real(S))
        print(S2)
        y_array[i] = S2
        if new_n ** 2 - 2 * new_n == 0:
            new_p = new_p
        else:
            new_p = new_p * new_n / (new_n - 2) - 2 * c / ((new_n - 1) * (new_n - 2))
            new_n -= 1

    return x_array, y_array


def s_sim(n, p, smoothing=False): # i think this is for little s - the average small component size
    y_array = np.zeros(n)
    x_array = np.zeros(n)
    new_p = p
    new_n = n
    for i in range(n):
        x_array[i] = i / n

    for i in range(n - 2):
        c = 2 * new_p * scipy.special.comb(new_n, 2) / new_n
        # print(c)
        S = 1 + scipy.special.lambertw((-c * np.exp(-c)), k=0, tol=1e-8) / c
        # <s> = 1/(1-c+cS)
        y_array[i] = 1 / (1 - c + c * S)
        # print(y_array[i])
        if new_n ** 2 - 2 * new_n == 0:
            new_p = new_p
        else:
            new_p = new_p * (new_n ** 2 - 2 * new_n + 4) / (new_n ** 2 - 2 * new_n)
        new_n -= 1

    return x_array, y_array


def s_sim_attack(n, p, smoothing=False): # average small component size for attack removals
    y_array = np.zeros(n)
    x_array = np.zeros(n)

    new_p = p
    new_n = n
    for i in range(n):
        x_array[i] = i / n

    for i in range(n - 2):
        max = maxdegree(new_n, new_p)
        c = 2 * new_p * scipy.special.comb(new_n, 2) / new_n
        # print(c)
        S = 1 + scipy.special.lambertw((-c * np.exp(-c)), k=0, tol=1e-8) / c
        # <s> = 1/(1-c+cS)
        y_array[i] = 1 / (1 - c + c * S)
        # print(y_array[i])
        if new_n ** 2 - 2 * new_n == 0:
            new_p = new_p
        else:
            new_p = new_p * new_n / (new_n - 2) - 2 * max / ((new_n - 1) * (new_n - 2))
            # - 2*max/((new_n-1)(new_n-2))
        new_n -= 1

    return x_array, y_array

# takes in parameters for graph type, number of nodes, number of edges per node,
# remove strategies, and performance metrics
# helper function for plot_graphs

def completeRobustnessData(graph_types=['ER', 'SF'],
                           numbers_of_nodes=np.arange(20, 101, 20),
                           # 20,101,20: i.e., 20, 40 ,60, 80, 100 nodes
                           numbers_of_edges=[1, 2, 3],
                           remove_strategies=['random', 'attack'],
                           performance='efficiency'):
    # initialize some big list
    # in the lambert/recursion stuff, don't actually use all these at once. But maybe in the future?
    LIST = [[[[0 for i in range(len(remove_strategies))]
              for j in range(len(numbers_of_edges))]
             for k in range(len(numbers_of_nodes))]
            for l in range(len(graph_types))]

    for i_gt, graph_type in enumerate(graph_types):
        # enumerate() indexes each graph_type in graph_types
        for i_nn, number_of_nodes in enumerate(numbers_of_nodes):
            for i_ne, number_of_edges in enumerate(numbers_of_edges):
                for i_rs, remove_strategy in enumerate(remove_strategies):
                    # for every graph size, average degree, and removal strategy, get the data of the performance
                    # values over all the trials (10 is the default)
                    data = computeRobustnessCurves(graph_type=graph_type,
                                                   number_of_nodes=number_of_nodes,
                                                   number_of_edges=number_of_edges, remove_nodes=remove_strategy,
                                                   performance=performance)[0]
                    LIST[i_gt][i_nn][i_ne][i_rs] = data #put this data into the list
    return LIST

    # returns 6-dimensional array indicating performance over removing
    # n nodes, num_trials times,

#calculated = big_S(.1,[200],fvalues,pvalues)
#print("calc")
# print(calculated[0])
# print(calculated[1])
# print("calcend")
#plt.plot(np.flip(np.arange(100)/100), calculated / 100, label="calculated")
#plt.show()

#plt.plot(np.flip(np.arange(100)), calculated / 100,
                                     #label="calculated")
#plt.plot(perf_sim2copy(100, .1, smoothing=False)[0],
                                     #perf_sim2copy(100, .1, smoothing=False)[2], label='Theoretical')
#plt.show()


# WORKS
def plot_graphs(graph_types=['ER', 'SF'],
                numbers_of_nodes=[100],  # 20,101,20: i.e., 20, 40 ,60, 80, 100 nodes
                numbers_of_edges=[1, 2, 3],
                remove_strategies=['random', 'attack'],
                performance='efficiency', to_vary='nodes', vary_index=1, smoothing=False, both=False,
                forbidden_values=[]):  # figure out what vary_index did again
    LIST = completeRobustnessData(graph_types, numbers_of_nodes, numbers_of_edges, remove_strategies, performance)

    if to_vary == 'nodes': # I use this one mostly - it means that we're comparing by graph size
        # start plotting
        fig = plt.figure()

        # 1. plot performance as function of n
        x = 1
        while x < 2: # since only 1 subplot right now, set while x<2
            for i_gt, graph_type in enumerate(graph_types):
                for i_rs, remove_strategy in enumerate(remove_strategies):

                    ax1 = plt.subplot(2, 2, x)  # 1 subplot for every combination of graph type/rs
                                                # only 1 subplot right now
                    x += 1
                    print("x")
                    for i_nn, number_of_node in enumerate(numbers_of_nodes):
                        # percolation threshold stuff from the robustness graphs - vertical line
                        # if computeRobustnessCurves(graph_type=graph_type,
                        #                          number_of_nodes=number_of_node, remove_nodes = remove_strategy,
                        #                          number_of_edges = numbers_of_edges[0],
                        #                          performance=performance)[1] > 0:
                        #   percolation_threshold_graph = 1 - computeRobustnessCurves(graph_type=graph_type, remove_nodes = remove_strategy,
                        #                          number_of_nodes=number_of_node,
                        #                          number_of_edges = numbers_of_edges[0],
                        #                          performance=performance)[1]
                        #   print("pcg", 1-percolation_threshold_graph)
                        # generating plot
                        # x_points = computeRobustnessCurves(graph_type=graph_type,
                        #                          number_of_nodes=number_of_node,
                        #                          number_of_edges = numbers_of_edges[0],
                        #                          performance=performance)[2]

                        # calculating p from the m and n values in the parameters - when I put
                        # p=.1 in the construct_a_network function and below, this line gets overwritten
                        p = 2 * numbers_of_edges[i_nn] * (number_of_node - numbers_of_edges[i_nn]) / (
                                    number_of_node * (number_of_node - 1))

                        # don't know why we have this
                        calc_array = np.flip(np.arange(number_of_node) + 1)
                        #print("calc_array")
                        #calculated = big_S(p, np.flip(np.arange(number_of_node)),fvalues,pvalues)

                        # calculate the rel LCC from recursion for p=.1, and the graph size in question
                        calculated = big_relS(.1,[number_of_node],fvalues,pvalues)
                        print(calculated)
                        #print(calculated) # checkpoints
                        #print("calculated")
                        x_points = np.zeros(number_of_node)
                        for u in range(number_of_node):
                            x_points[u] = u / number_of_node
                        #print("xpoints")

                        #get simulated data from the big list
                        data_array = np.array(LIST[i_gt][i_nn][vary_index][i_rs][1:])
                        # print("data array shape", data_array.shape)

                        #this prevented some sort of bug about invalid values
                        for val in forbidden_values:
                            data_array[data_array == val] = np.nan
                        # print("data array shape", data_array.shape)

                        # print(np.nanmean(data_array,
                        # axis = 0))
                        #plot the simulated data
                        ax1.plot(x_points,  # range(numbers_of_nodes[i_nn]),
                                 np.nanmean(data_array,
                                            axis=0), 'o-',
                                 label="number of nodes=" + str(numbers_of_nodes[i_nn]))
                        print(np.nanmean(data_array,
                                            axis=0))
                        #print("nanmean")
                        # plt.show()

                        # simulation
                        #p = 2 * numbers_of_edges[0] * (numbers_of_nodes[0] - numbers_of_edges[0]) / (
                                    #numbers_of_nodes[0] * (numbers_of_nodes[0] - 1))

                        p=.1 # I guess I also defined p here too for the lambert function
                        if performance == "relative LCC" and remove_strategies == ["random"]:
                            # the commented lines are from other notebooks where I was plotting different things, like
                            # lambert functions while keeping the ratio of p to graph size constant, etc.

                            # ax1.plot(rich(p, fixed=.8)[0],rich(p, fixed=.8)[1])
                            # ax1.plot(perf_sim2copy(numbers_of_nodes[0], p, smoothing = smoothing)[0], perf_sim2copy(numbers_of_nodes[0], p, smoothing = smoothing)[1], label = 'Lambert Input')

                            #lambert plot
                            ax1.plot(perf_sim2copy(numbers_of_nodes[0], p, smoothing=smoothing)[0],
                                     perf_sim2copy(numbers_of_nodes[0], p, smoothing=smoothing)[2], label='Theoretical')
                            #recursive plot
                            ax1.plot(np.flip(np.arange(0,number_of_node))/number_of_node, calculated,
                                     label="calculated")
                            print(calculated) #checkpoint
                            plt.show()
                            ax1.legend()

                            # ax1.plot(perf_sim2copy(numbers_of_nodes[0], p, smoothing = smoothing)[0], perf_sim2copy(numbers_of_nodes[0], p, smoothing = smoothing)[3], label = 'mean degree')
                            # ax1.plot(perf_sim2copy(numbers_of_nodes[0], p, smoothing = smoothing)[0], perf_sim2copy(numbers_of_nodes[0], p, smoothing = smoothing)[4], label = 'difference')
                            # ax1.plot(np.linspace(0,1,100), -1/np.exp(1)*np.ones(shape=(100)))
                            # ax1.plot(np.linspace(0,1,100), np.ones(shape=(100)))
                            # plt.xlim(.4, .6)
                            # plt.ylim(-.5,1.2)

                            # ax1.plot(perf_sim_revision1(numbers_of_nodes[0], p)[0], perf_sim_revision1(numbers_of_nodes[0], p)[1])
                            print("work")
                        if performance == "relative LCC" and remove_strategies == ["attack"]:
                            ax1.plot(perf_sim2_attack(numbers_of_nodes[0], p, smoothing=smoothing)[0],
                                     perf_sim2_attack(numbers_of_nodes[0], p, smoothing=smoothing)[1])

                        if performance == "average small component size" and remove_strategies == ["random"]:
                            if both == True:
                                ax1.plot(perf_sim2(numbers_of_nodes[0], p, smoothing=smoothing)[0],
                                         perf_sim2(numbers_of_nodes[0], p, smoothing=smoothing)[1])
                            ax1.plot(s_sim(numbers_of_nodes[0], p)[0], s_sim(numbers_of_nodes[0], p)[1])
                            plt.ylim(0, 5)

                        if performance == "average small component size" and remove_strategies == ["attack"]:
                            ax1.plot(s_sim_attack(numbers_of_nodes[0], p)[0], s_sim_attack(numbers_of_nodes[0], p)[1])
                            plt.ylim(0, 5)

                        if performance == "both" and remove_strategies == ["random"]:
                            ax1.plot(perf_sim2(numbers_of_nodes[0], p, smoothing=smoothing)[0],
                                     perf_sim2(numbers_of_nodes[0], p, smoothing=smoothing)[1])
                            ax1.plot(s_sim(numbers_of_nodes[0], p)[0], s_sim(numbers_of_nodes[0], p)[1])
                            plt.ylim(0, 5)

                        # print(range(numbers_of_nodes[i_nn]))
                        # print(np.nanmean(LIST[i_gt][i_nn][vary_index][i_rs][1:], axis = 0))
                        # labeling plot
                        ax1.set_title(str(performance) + " of " + str(graph_type) +
                                      " graph, " + str(remove_strategy) + " removal: over "
                                      + to_vary)
                        ax1.legend()
                        ax1.set_xlabel('n (number nodes removed)')
                        ax1.set_ylabel(performance)

                        # this was from robustness library stuff/plotting different percolation thresholds
                        #for different sized graphs.
                        if number_of_node == numbers_of_nodes[0]:
                            set_color = "tab:blue"
                        # if number_of_node == numbers_of_nodes[1]:
                        #   set_color = "tab:orange"
                        # if number_of_node == numbers_of_nodes[2]:
                        #   set_color = "tab:green"
                        # if number_of_node == numbers_of_nodes[3]:
                        #   set_color = "tab:red"
                        # if number_of_node == numbers_of_nodes[4]:
                        #   set_color = "tab:purple"
                        #laterpercolation_threshold_graph = 1 - perf_sim2copy(numbers_of_nodes[0], p, smoothing=smoothing)[5]
                        # print(percolation_threshold_graph)
                        #laterax1.axvline(percolation_threshold_graph, color=set_color)

                    # fig.tight_layout()
            plt.subplots_adjust(left=None, bottom=None, right=2, top=2, wspace=None, hspace=None)
            plt.show()

    # this isn't used so much now - it was mostly from robustness stuff when I would compare
    # performance values when there were different densities
    elif to_vary == 'edges':
        # start plotting
        fig = plt.figure()

        # 1. plot performance as function of n, 2 * 2 * 5 figures
        x = 1
        while x < 5:
            for i_gt, graph_type in enumerate(graph_types):
                for i_rs, remove_strategy in enumerate(remove_strategies):
                    ax1 = plt.subplot(2, 2, x)  # 1 subplot for every combination of graph type/rs
                    x += 1

                    for i_ne, number_of_edge in enumerate(numbers_of_edges):
                        # generating plot
                        ax1.plot(range(numbers_of_nodes[1]),
                                 np.mean(LIST[i_gt][vary_index][i_ne][i_rs][1:],
                                         axis=0), 'o-',
                                 label="number of edges=" + str(numbers_of_edges[i_ne]))

                        # labeling plot
                        ax1.set_title(str(performance) + " of " + str(graph_type) +
                                      " graph, " + str(remove_strategy) + " removal: over "
                                      + to_vary)
                        ax1.legend()
                        ax1.set_xlabel('n (number nodes removed)')
                        ax1.set_ylabel(performance)
                        # plt.show
                    # fig.tight_layout()
            plt.subplots_adjust(left=None, bottom=None, right=2, top=2, wspace=None, hspace=None)
            plt.show()

        else:
            print("Please vary either nodes or edges.")
            # safety check if user made a mistake with to_vary

#plot_graphs(['ER'], [2], [1], ["random"], performance = 'relative LCC', vary_index = 0, smoothing = True, forbidden_values = [0])
#plt.show()

plot_graphs(['ER'], [20], [1], ["random"], performance = 'relative LCC', vary_index = 0, smoothing = True, forbidden_values = [0])
plt.show()

# print("calc f")
# print(calculate_f(.1,3,3,fvalues))
# print(calculate_f(.1,2,3,fvalues))
# print(calculate_f(.1,1,3,fvalues))
# print(calculate_f(.2,3,3,fvalues))
# print(calculate_f(.2,2,3,fvalues))
# print(calculate_f(.2,1,3,fvalues))
# print("calc P")
# print(calculate_P(.1, 3, 3, fdict=fvalues, pdict=pvalues))
# print(calculate_P(.1, 2, 3, fdict=fvalues, pdict=pvalues))
# print(calculate_P(.1, 1, 3, fdict=fvalues, pdict=pvalues))
# print("calcS")
# print(calculate_S(.1,3,fvalues,pvalues))