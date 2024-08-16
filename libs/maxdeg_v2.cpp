#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <omp.h> // OpenMP for parallelism

std::unordered_map<int, int> binomial_coeff_memo;
std::unordered_map<std::pair<int, int>, double, boost::hash<std::pair<int, int>>> binomial_pmf_memo;
std::mutex mtx;

// Function to calculate binomial coefficient with memoization
int binomial_coeff(int n, int k) {
    int key = n * 1000 + k; // Create a unique key for n and k
    if (binomial_coeff_memo.find(key) != binomial_coeff_memo.end())
        return binomial_coeff_memo[key];

    if (k > n) return 0;
    if (k == 0 || k == n) return 1;

    int res = 1;
    for (int i = 0; i < k; ++i) {
        res *= (n - i);
        res /= (i + 1);
    }
    binomial_coeff_memo[key] = res;
    return res;
}

// Function to calculate the binomial probability mass function with memoization
double binomial_pmf(int k, int n, double p) {
    std::pair<int, int> key = std::make_pair(k, n);
    if (binomial_pmf_memo.find(key) != binomial_pmf_memo.end())
        return binomial_pmf_memo[key];

    double result = binomial_coeff(n, k) * std::pow(p, k) * std::pow(1 - p, n - k);
    binomial_pmf_memo[key] = result;
    return result;
}

// Function to compute pair approximation probability with precomputation and parallelism
double pair_approximation_probability(const std::vector<int>& k_values, int n, int ell) {
    int m = k_values.size();
    double probability_sum = 0.0;

    #pragma omp parallel for reduction(+:probability_sum)
    for (int mask = 0; mask < (1 << m); ++mask) {
        if (__builtin_popcount(mask) != ell) continue;

        double prod_1 = 1.0;
        double prod_2 = 1.0;
        for (int j = 0; j < m; ++j) {
            if (mask & (1 << j)) {
                prod_1 *= k_values[j] / static_cast<double>(n - 1 - j);
            } else {
                prod_2 *= (1 - k_values[j] / static_cast<double>(n - 1 - j));
            }
        }
        probability_sum += prod_1 * prod_2;
    }

    return probability_sum;
}

// Function to compute p(k_i = kappa_i) for a given degree sequence with precomputation and parallelism
double p_k_i_given_degree_sequence(const std::vector<int>& k_values, int n, double p, int i, int kappa_i) {
    double probability = 0.0;

    #pragma omp parallel for reduction(+:probability)
    for (int ell = 0; ell < i; ++ell) {
        double prob_ell = pair_approximation_probability(k_values, n, ell);
        double binomial_prob = binomial_pmf(kappa_i - ell, n - i, p);
        probability += prob_ell * binomial_prob;
    }

    return probability;
}

// Function to compute the joint probability distribution for the entire degree sequence with parallelism
double joint_probability_distribution(const std::vector<int>& degree_sequence, int n, double p) {
    double joint_probability = 1.0;

    #pragma omp parallel for reduction(*:joint_probability)
    for (int i = 1; i <= degree_sequence.size(); ++i) {
        int kappa_i = degree_sequence[i - 1];
        std::vector<int> k_values(degree_sequence.begin(), degree_sequence.begin() + i - 1);
        double prob = p_k_i_given_degree_sequence(k_values, n, p, i, kappa_i);
        joint_probability *= prob;
    }

    return joint_probability;
}

// Function to calculate the expected maximum degree in a random graph with parallelism
double expected_maximum_degree(int n, double p) {
    double max_degree_expectation = 0.0;
    int possible_degrees = n;

    #pragma omp parallel for reduction(+:max_degree_expectation)
    for (int mask = 0; mask < std::pow(possible_degrees, n); ++mask) {
        std::vector<int> degree_sequence(n);
        int temp_mask = mask;

        for (int i = 0; i < n; ++i) {
            degree_sequence[i] = temp_mask % possible_degrees;
            temp_mask /= possible_degrees;
        }

        double joint_prob = joint_probability_distribution(degree_sequence, n, p);
        int max_degree = *std::max_element(degree_sequence.begin(), degree_sequence.end());
        max_degree_expectation += max_degree * joint_prob;
    }

    return max_degree_expectation;
}

int main() {
    int n = 5;         // Example total number of nodes (small for computational feasibility)
    double p = 0.5;    // Example probability of edge between any two nodes

    double expected_max_degree = expected_maximum_degree(n, p);
    std::cout << "Expected maximum degree in the random graph: " << expected_max_degree << std::endl;

    return 0;
}
