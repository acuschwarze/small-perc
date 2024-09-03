#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <numeric>

// TODO: add more node removals and return degree distributions in a useful format

// Function to calculate the degree sequence of a graph
std::vector<int> calculate_degree_sequence(int n, const std::vector<int>& graph) {
    std::vector<int> degree(n, 0);
    int index = 0;

    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (graph[index++] == 1) {
                degree[i]++;
                degree[j]++;
            }
        }
    }

    // std::cout << "ds ";
    // for (double value : degree) {
    //    std::cout << value << " ";
    // }
    // std::cout << std::endl;

    return degree;
}

// Function to calculate the degree distribution of a graph
std::vector<int> calculate_degree_distribution(int n, const std::vector<int>& degree) {

    // Calculate the degree distribution (number of nodes for each degree)
    std::vector<int> degree_distribution(n, 0.0);
    for (int d : degree) {
        degree_distribution[d]++;
    }

    // std::cout << "dd ";
    // for (double value : degree_distribution) {
    //    std::cout << value << " ";
    // }
    // std::cout << std::endl;

    // double sum = std::accumulate(degree_distribution.begin(), degree_distribution.end(), 0.0);

    // Step 2: Divide each element by the sum
    // for (int& element : degree_distribution) {
    //    element /= sum;
    // }

    return degree_distribution;
}

// Function to remove the highest degree vertex and recalculate the degree distribution
std::vector<int> remove_highest_degree_vertex(int n, const std::vector<int>& degree, const std::vector<int>& graph) {
    int max_degree_index = std::max_element(degree.begin(), degree.end()) - degree.begin();
    std::vector<int> new_degree(degree);

    int index = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (graph[index++] == 1) {
                if (i == max_degree_index || j == max_degree_index) {
                    new_degree[i]--;
                    new_degree[j]--;
                }
            }
        }
    }
    new_degree.erase(new_degree.begin() + max_degree_index);

    // std::cout << "new ds ";
    // for (double value : new_degree) {
    //     std::cout << value << " ";
    // }
    // std::cout << std::endl;

    return new_degree;
}

// Function to calculate the expected degree distributions
void calculate_expected_distributions(int n, double p, std::vector<double>& expected_degree_distribution, std::vector<double>& expected_distribution_after_removal) {
    int total_edges = n * (n - 1) / 2;
    int num_graphs = 1 << total_edges; // 2^(n*(n-1)/2) graphs
    double sum_probabilities = 0.0;

    for (int i = 0; i < num_graphs; ++i) {
        std::vector<int> graph(total_edges);

        // Generate the graph
        for (int j = 0; j < total_edges; ++j) {
            graph[j] = (i >> j) & 1;
        }

        int edges = std::count(graph.begin(), graph.end(), 1);
        double probability = std::pow(p, edges) * std::pow(1 - p, total_edges - edges);

        std::vector<int> degree_sequence = calculate_degree_sequence(n, graph);
        std::vector<int> degree_distribution = calculate_degree_distribution(n, degree_sequence);
        // std::cout << "removed node" << std::endl;
        std::vector<int> sequence_after_removal = remove_highest_degree_vertex(n, degree_sequence, graph);
        std::vector<int> distribution_after_removal = calculate_degree_distribution(n-1, sequence_after_removal);

        sum_probabilities += probability;
        // std::cout << "probability " << probability << std::endl;
        // std::cout << "degree_distribution ";
        // for (double value : degree_distribution) {
        //     std::cout << value << " ";
        // }
        // std::cout << std::endl;

        for (int k = 0; k < n; ++k) {
            expected_degree_distribution[k] += degree_distribution[k] * probability;
        }

        for (int k = 0; k < n - 1; ++k) {
            expected_distribution_after_removal[k] += distribution_after_removal[k] * probability;
        }
    }

    std::cout << "sum_probabilities " << sum_probabilities << std::endl;

}

int distribution_after_node_removal() {
    int n = 8; // Number of nodes
    double p = 0.2; // Probability of edge inclusion

    std::vector<double> expected_degree_distribution(n, 0.0);
    std::vector<double> expected_distribution_after_removal(n - 1, 0.0);

    // Calculate the expected degree distributions
    calculate_expected_distributions(n, p, expected_degree_distribution, expected_distribution_after_removal);

    // Output the expected degree distribution
    std::cout << "Expected Degree Distribution: ";
    for (double value : expected_degree_distribution) {
        std::cout << value << " ";
    }
    std::cout << std::endl;

    // Output the expected degree distribution after vertex removal
    std::cout << "Expected Degree Distribution After Vertex Removal: ";
    for (double value : expected_distribution_after_removal) {
        std::cout << value << " ";
    }
    std::cout << std::endl;

    return 0;
}

int main() {
    // Record start time
    auto start = std::chrono::high_resolution_clock::now();

    // Call the function you want to time
    distribution_after_node_removal();

    // Record end time
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate the duration
    std::chrono::duration<double> duration = end - start;

    // Output the time taken
    std::cout << "Time taken by function: " << duration.count() << " seconds" << std::endl;

    return 0;
}