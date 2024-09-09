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
    double b = 0;
    double for(int i = 0; int i < k+1, int i++) {
        b = b+binomial(n,i,p);
    }
    return b
}


double probs_less(int n, double p, int k): //probability that in a G(n,p), all nodes have degree < k
    //print("n p k",n,p,k)
    if (k == 0) {
        double prob = 0;
    } else if (k == n-1) {
        double prob = 1;
    } else if (n == 1) {
        double prob = 1;
    } else {
        double k_possibilities = 0;
        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < n; ++j)= { // j for all values of exactly how many nodes in n-1 have less than k-1 degree
                double x = 0;
                for (int i_x = 0; i_x < j; ++i_x) {
                    //x += 1-((1-binomialDistribution.cdf(i_x,n-1,p))**(i_x)*scipy.special.comb(n-1,i_x)) # probability that there are at least j nodes with less than k-1 
                    x += pow(cdf(n-1,k-2,p) , i_x) * comb(n-1,i_x);
                }
                k_possibilities += probs_less(n-1,p,i) * comb(j,i) * pow(p,i) * pow((1-p) , (n-1-i)) * (x);
            //k_possibilities += sum / i
        // (binomialDistribution.cdf(k-1, n - 1, p))**i
            } 
        }
        prob = k_possibilities;
    }
    
    return prob

double expectedMaxDegree(int n, double p):
    if (n in [0, 1]) or (p == 0) {
        return 0;
    }
    
    double array[n+1];
    for (int j = 0; j < n; ++j) {
        array[j] = 1-probs_less(n,p,j);
    }
    array[n] = 0;
    double probs_kmax[n] = array[:-1] - array[1:];
    double mean_k_max = 0;
    for (int i_m = 0; i_m < n; ++i_m) {
        mean_k_max = mean_k_max + probs_kmax[i_m]*i_m;
    }
    //("probs_kmax",probs_kmax)
    //print("meankmax",mean_k_max)
    return mean_k_max




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