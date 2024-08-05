// cppimport
#include <pybind11/pybind11.h>
#include <cmath>
#include <iostream>
#include <unordered_map>

namespace py = pybind11;

// Function to calculate the factorial
int factorial(int n) {
    if (n <= 1)
        return 1;
    return n * factorial(n - 1);
}

// Function to calculate the combination
double comb(int n, int r) {
    return factorial(n) / (factorial(r) * factorial(n - r));
}

//Calculate g(p,i,n) for calculate_P_mult
double calculate_g(double p, int i, int n) {
    double g = pow(1 - p, i * (n - i));
    return g;
}

//Calculate f(p,i,n) for calculate_P_mult
double calculate_f(double p, int i, int n, std::unordered_map<double, std::unordered_map<int, std::unordered_map<int, double>>>& fdict) {
    if (fdict.find(p) != fdict.end() && fdict[p].find(n) != fdict[p].end() && fdict[p][n].find(i) != fdict[p][n].end()) {
        return fdict[p][n][i];
    } else {
        double sum_f = 0;
        for (int i_n = 1; i_n < i; ++i_n) {
            sum_f += (calculate_f(p, i_n, n, fdict) * comb(i - 1, i_n - 1) * pow(1 - p, i_n * (i - i_n)));
        }
        double f = 1 - sum_f;
        fdict[p][n][i] = f; // does that change the original?
        return f;
    }
}

// Calculate h(p,i,n,k) for calculate_P_mult
double calculate_h(double p, int i, int n, int k, std::unordered_map<double, std::unordered_map<int, std::unordered_map<int, double>>>& fdict) {
    double h = comb((n - (k - 1) * i), i) * calculate_f(p, i, n, fdict) * calculate_g(p, i, (n - (k - 1) * i));
    return h;
}

// Calculate probability of a G(n,p) graph to have a largest connected component with i nodes. Calculation accounts for the possibility of multiple largest connected components.
double calculate_P_mult(double p, int i, int n, std::unordered_map<double, std::unordered_map<int, std::unordered_map<int, double>>>& fdict, std::unordered_map<double, std::unordered_map<int, std::unordered_map<int, double>>>& pdict) {
    if (i == 1 && n == 1) {
        return 1;
    } else if (i == 1 && n != 1) {
        return pow(1 - p, comb(n, 2));
    } else {
        double P_tot = 0;
        for (int k = 1; k <= n / i; ++k) {
            double product = 1;
            for (int k_2 = 1; k_2 <= k; ++k_2) {
                product *= alice_helper(p, i, n, k_2, fdict);
            }

            double sum_less = 0;
            for (int j = 1; j < i; ++j) {
                sum_less += alice(p, j, n - k * i, fdict, pdict);
            }

            P_tot += 1.0 / factorial(k) * product * sum_less;
        }
        return P_tot;
    }
}


#include <cmath>
#include <stdio.h>

double binomial(int n, int k, double p){
    double p; 
    int k;
    int n;

    scanf("%d%d", &n, &k);
    scanf("%lf", &p);

    if (k > n) return 1;
    if (p > 1 || p < 0) return 1;

    double w = 1;   //neutral element of multiplication

    // n choose k part
    for (int i = n - k + 1; i <= n; ++i) w = w * i;
    for (int i = 1; i <= k; ++i) w = w / i;

    // p^k * (1-p)^(n-k) part
    w = w * pow(p, k) * pow(1.0 - p, n - k);

    printf("%lf\n", w);
    return w;
}

double cdf(int n, int k, double p){
    double b = 0
    double for(int i = 0; int i < k+1, int i++) {
        b = b+binomial(n,i,p)
    }
    return b
}



double expectedMaxDegree(int n, double p) {
    if (n in [0, 1] or p == 0) {
        return 0
    }
    if (n == 2) {
        return p
    }
    int k_max = 0

    xt::xarray<double> probs_k_or_less = xt::xarray<double>(n)

    for(int i = 0; int i < n; int i++) {
        probs_k_or_less[i] == cdf(n-1,i,p)
    }
    
    xt::xarray<double> probs_at_least_k = xt::concatenate(xtuple({1}, , c), 1)
    probs_at_least_k = np.concatenate([[1], np.array(1 - probs_k_or_less[:-1])])
    probs_at_least_k = np.cumsum([binomialDistribution.pmf(k, n - 1, p) for k in range(n)][::-1])[::-1]
    probs_at_least_one_node = 1 - (1 - probs_at_least_k) ** (n - k_max)

    # every node has at least degree zero
    #probs_at_least_one_node[0] = 1
    # at least one node has degree 1 if the graph is not empty
    #probs_at_least_one_node[1] = 1 - binomialDistribution.pmf(0, n * (n - 1) / 2, p)

    probs_at_least_one_node = np.concatenate([probs_at_least_one_node, [0]])
    probs_kmax = probs_at_least_one_node[:-1] - probs_at_least_one_node[1:]
    mean_k_max = np.sum([probs_kmax[k] * k for k in range(n)])

    return mean_k_max
}


def edgeProbabilityAfterTargetedAttack(n, p):
    '''Calculate edge probability in an Erdos--Renyi network with original size
    `n` and original edge probability `p` after removing the node with the
    highest degree.

    Parameters
    ----------
    n : int
       Number of nodes.
    
    p : float
       Edge probability in Erdos Renyi graph.
       
    Returns
    -------
    new_p (float)
       Updated edge probability.
    '''
    if n <=2:
        new_p = 0

    else:
        emd = expectedMaxDegree(n, p)
        new_p = p * n / (n - 2) - 2 * emd / ((n - 1) * (n - 2))
        new_p = max([new_p, 0])

    return new_p











int main() {
    // Example usage
    std::unordered_map<double, std::unordered_map<int, std::unordered_map<int, double>>> fdict, pdict;
    double result = calculate_P_mult(0.5, 3, 5, fdict, pdict);
    std::cout << "Result: " << result << std::endl;
    return 0;
}


PYBIND11_MODULE(recursion, m) {
    m.doc() = "submodule for fast recursive calculation of probabilities in c++"; 

    m.def("calculate_P_full", &calculate_P_full, "A function that calculates probability of a G(n,p) graph to have a largest connected component with i nodes.");
}