#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <numeric>
#include <iomanip>
#include <sstream>

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
    return degree;
}

// Function to calculate the degree distribution of a graph
std::vector<int> calculate_degree_distribution(int n, const std::vector<int>& degree) {
    std::vector<int> degree_distribution(n, 0.0);
    for (int d : degree) {
        degree_distribution[d]++;
    }
    return degree_distribution;
}

// Function to remove the highest degree vertex and recalculate the degree distribution
std::vector<int> remove_highest_degree_vertex(int n, const std::vector<int>& degree, const std::vector<int>& graph) {
    int max_degree_index = std::max_element(degree.begin(), degree.end()) - degree.begin();
    std::vector<int> new_graph;

    int index = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (i != max_degree_index && j != max_degree_index) {
                new_graph.push_back(graph[index]);
            }
            index++;
        }
    }
    return new_graph;
}

// Function to generate the lists of doubles
std::vector<std::vector<double>> calculate_expected_distributions(int n, double p) {
    int total_edges = n * (n - 1) / 2;
    int num_graphs = 1 << total_edges; // 2^(n*(n-1)/2) graphs
    double sum_probabilities = 0.0;

    // initialize arrays
    std::vector<std::vector<int>> expected_degree_sequences;
    std::vector<std::vector<double>> expected_degree_distributions;
    for (int i = 0; i < n - 1; ++i) {
        int size = n - i;
        std::vector<int> degree_sequence(size);
        std::vector<double> degree_distribution(size);
        expected_degree_sequences.push_back(degree_sequence);
        expected_degree_distributions.push_back(degree_distribution);
    }

    for (int i = 0; i < num_graphs; ++i) {
        std::vector<int> graph(total_edges);
        std::vector<std::vector<int>> subgraphs;

        // Generate the graph
        for (int j = 0; j < total_edges; ++j) {
            graph[j] = (i >> j) & 1;
        }
        subgraphs.push_back(graph);

        int edges = std::count(graph.begin(), graph.end(), 1);
        double probability = std::pow(p, edges) * std::pow(1 - p, total_edges - edges);
        sum_probabilities += probability;

        for (int k = 0; k < n-1; k++) {
            std::vector<int> current_degree_sequence = 
                calculate_degree_sequence(n-k, subgraphs[k]);
            std::vector<int> current_degree_distribution = 
                calculate_degree_distribution(n-k, current_degree_sequence);

            for (int l = 0; l < n-k; ++l) {
                expected_degree_distributions[k][l] += current_degree_distribution[l] * probability;
            }
            subgraphs.push_back(remove_highest_degree_vertex(n, current_degree_sequence, subgraphs[k]));
        }
    }

    return expected_degree_distributions;
}

// Function to write lists to console output
void write_to_console(const std::vector<std::vector<double>>& lists) {
    // Write each list to console
    for (const auto& list : lists) {
        for (size_t i = 0; i < list.size(); ++i) {
            std::cout << std::fixed << std::setprecision(6) << list[i];
            if (i != list.size() - 1) {
                std::cout << " ";
            }
        }
        std::cout << std::endl;
    }
}

int main(int argc, char* argv[]) {
    // Default values
    int n = 6;
    double p = 0.2;

    // Parse command line arguments if provided
    if (argc == 3) {
        n = std::stoi(argv[1]);
        p = std::stod(argv[2]);
    }

    // Generate the lists of doubles
    std::vector<std::vector<double>> distributions = calculate_expected_distributions(n, p);

    // Write the lists to console
    write_to_console(distributions);

    return 0;
}