//#include <pybind11/pybind11.h>
#include <cmath>
#include <iostream>
#include <unordered_map>
#include <algorithm>
#include <vector>
//#include <quadmath.h> // for double
#include <cstdlib> // for command line input
#include <string> // for std::string
#include <sstream> // for std::stringstream
#include <chrono> // for timing

// cppimport
//#include <pybind11/pybind11.h>
#include <cmath>
#include <iostream>
#include <unordered_map>

//namespace py = pybind11;

__float128 factorial(int n) {
    if (n <= 1) 
        return 1.0;
    return n * factorial(n - 1);
}

// Function to calculate the combination
__float128 comb(long long n, long long r) 
{
    __float128 f = 1; 
    for(auto i = 0; i < r;i++)
        f = (f * (n - i)) / (i + 1);
    return f ; 
}

// Function to calculate the factorial
// int factorial(int n) {
//     if (n <= 1)
//         return 1;
//     return n * factorial(n - 1);
// }

// // Function to calculate the combination
// double comb(int n, int r) {
//     return factorial(n) / (factorial(r) * factorial(n - r));
// }

#include <cmath>
#include <stdio.h>

double binomial(int n, int k, double p){
    std::cout << "binomial";

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
    std::cout << "cdf";
    double b = 0;
    for(int i = 0; i < k+1; i++) b = b+binomial(n,i,p);
    return b;
}


__float128 probs_less(int n, double p, int k){
    std::cout << "probsless";
//probability that in a G(n,p), all nodes have degree < k
    //print("n p k",n,p,k)
    if (k == 0) {
        return 0;
    } else if (k == n-1) {
        return 1;
    } else if (n == 1) {
        return 1;
    } else {
        __float128 k_possibilities = 0;
        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < n; ++j) { // j for all values of exactly how many nodes in n-1 have less than k-1 degree
                __float128 x = 0;
                for (int i_x = 0; i_x < j; ++i_x) {
                    //x += 1-((1-binomialDistribution.cdf(i_x,n-1,p))**(i_x)*scipy.special.comb(n-1,i_x)) # probability that there are at least j nodes with less than k-1 
                    x += pow(cdf(n-1,k-2,p) , i_x) * comb(n-1,i_x);
                }
                k_possibilities += probs_less(n-1,p,i) * comb(j,i) * pow(p,i) * pow((1-p) , (n-1-i)) * (x);
            //k_possibilities += sum / i
        // (binomialDistribution.cdf(k-1, n - 1, p))**i
            } 
        }
        return k_possibilities;
    }
    
    //return prob;
}

__float128 expectedMaxDegree(int n, double p) {
    std::cout << "expected max deg";
    if (n < 1) {
        return 0;
    } else if (p==0) {
        return 0;
    }
    std::vector< double > arr;
    for (size_t j = 0; j < n; ++j) {
        arr.push_back( 1-probs_less(n,p,j) );
    }
    arr.push_back(0);
    std::vector< double > probs_kmax;
    for (size_t i = 0; i < n; ++i) {
        probs_kmax.push_back(arr[i] - arr[i+1]);
        }
    __float128 mean_k_max = 0;
    for (size_t i_m = 0; i_m < n; ++i_m) {
        mean_k_max = mean_k_max + probs_kmax[i_m]*i_m;
    }
    //("probs_kmax",probs_kmax)
    //print("meankmax",mean_k_max)
    return mean_k_max;
}


__float128 edgeProbabilityAfterTargetedAttack(int n, double p) {
    // '''Calculate edge probability in an Erdos--Renyi network with original size
    // `n` and original edge probability `p` after removing the node with the
    // highest degree.

    // Parameters
    // ----------
    // n : int
    //    Number of nodes.
    
    // p : float
    //    Edge probability in Erdos Renyi graph.
       
    // Returns
    // -------
    // new_p (float)
    //    Updated edge probability.
    // '''
    std::cout << "edge prob after attack";
    if (n <=2) {
        return 0;
    } else {
        __float128 emd = expectedMaxDegree(n, p);
        __float128 new_p = p * n / (n - 2) - 2 * emd / ((n - 1) * (n - 2));
        new_p = fmax(new_p, 0);
        return new_p;
    }
}

int main(int argc, char* argv[]) {
    // Check if the correct number of command line arguments are provided
    // if (argc != 3) {
    //     std::cerr << "Usage: " << argv[0] << " n, p" << std::endl;
    //     return 1;
    // }

    // Parse command line arguments
    int n = std::atoi(argv[1]);
    double p = std::strtod(argv[2], nullptr);

    double result = edgeProbabilityAfterTargetedAttack(n,p);
    std::cout << "main";
    std::cout << result << std::endl;
}
