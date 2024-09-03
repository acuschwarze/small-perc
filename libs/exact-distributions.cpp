#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <numeric>
#include <fstream>
#include <string>
#include <iomanip>  // For setting precision
#include <sstream>  // For converting double to string
#include <sys/stat.h>
#include <sys/types.h>


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
    std::vector<int> new_graph;

    int index = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            // Skip all edges connected to the vertex with the highest degree
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

// Function to write lists to a file
void write_to_file(int n, double p, const std::vector<std::vector<double>>& lists) {
    // Create the output filename
    std::ostringstream filename;

        filename << "exact_degree_distributions_n" << n << "_p" << std::fixed << std::setprecision(2) << p << ".txt";

    // Open the output file
    std::ofstream output_file(filename.str());

    if (!output_file.is_open()) {
        std::cerr << "Failed to open the file: " << filename.str() << std::endl;
        return;
    }

    // Write each list to the file
    for (const auto& list : lists) {
        for (size_t i = 0; i < list.size(); ++i) {
            output_file << std::fixed << std::setprecision(6) << list[i];
            if (i != list.size() - 1) {
                output_file << " ";
            }
        }
        output_file << std::endl;
    }

    output_file.close();
}


int main(int argc, char* argv[]) {

    // Record start time
    auto start = std::chrono::high_resolution_clock::now();

    int n = 6;
    double p = 0.2;

    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <n> <p>" << std::endl;
        // return 1;
    }
    else {
        n = std::stoi(argv[1]);
        p = std::stod(argv[2]);
    }

    // Call the function you want to time
    // distribution_after_node_removal();

    // Generate the lists of doubles
    std::vector<std::vector<double>> distributions = calculate_expected_distributions(n, p);

    // Write the lists to the file
    write_to_file(n, p, distributions);

    std::cout << "Data written to file successfully." << std::endl;

    // Record end time
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate the duration
    std::chrono::duration<double> duration = end - start;

    // Output the time taken
    std::cout << "Time taken by function: " << duration.count() << " seconds" << std::endl;

    return 0;
}