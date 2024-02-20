//#include <pybind11/pybind11.h>
#include <cmath>
#include <iostream>
#include <unordered_map>
#include <cstdlib> // for command line input
#include <string> // for std::string
#include <sstream> // for std::stringstream
#include <chrono> // for timing

//namespace py = pybind11;

// Function to calculate the factorial
int factorial(int n) {
    if (n <= 1)
        return 1;
    return n * factorial(n - 1);
}

// Function to calculate the combination (old)
double comb_old(int n, int r) {
    std::cout << "comb" << n << r << std::endl;
    return factorial(n) / (factorial(r) * factorial(n - r));
}

// Function to calculate the combination
long long comb(long long n, long long r) 
{
    long long f = 1; 
    for(auto i = 0; i < r;i++)
        f = (f * (n - i)) / (i + 1);
    return f ; 
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
                product *= calculate_h(p, i, n, k_2, fdict);
            }

            double sum_less = 0;
            for (int j = 1; j < i; ++j) {
                sum_less += calculate_P_mult(p, j, n - k * i, fdict, pdict);
            }

            P_tot += 1.0 / factorial(k) * product * sum_less;
        }
        return P_tot;
    }
}

int timer() { //toggle to main if needed
    // Define input values for the loop
    int inputValues[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50};

    // Loop through each input value
    for (int i = 0; i < sizeof(inputValues) / sizeof(inputValues[0]); ++i) {
        // Get the current input value
        int inputValue = inputValues[i];

        // Timing setup
        auto startTime = std::chrono::high_resolution_clock::now();

        // Call the function
        std::unordered_map<double, std::unordered_map<int, std::unordered_map<int, double>>> fdict, pdict;
        double result = calculate_P_mult(0.2, inputValue, 2*inputValue, fdict, pdict);
    
        // Timing calculation
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);

        // Output timing information
        std::cout << "calculate_P_mult(0.2," << inputValue << "," << 2*inputValue << ") took " << duration.count() << " microseconds" << std::endl;
    }

    return 0;
}

int main(int argc, char* argv[]) {
    // Check if the correct number of command line arguments are provided
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " p, i, n" << std::endl;
        return 1;
    }

    // Parse command line arguments
    double p = std::strtod(argv[1], nullptr);
    int i = std::atoi(argv[2]);
    int n = std::atoi(argv[3]);
    std::unordered_map<double, std::unordered_map<int, std::unordered_map<int, double>>> fdict, pdict;
    double result = calculate_P_mult(p, i, n, fdict, pdict);
    std::cout << result << std::endl;
}


//PYBIND11_MODULE(recursion, m) {
//    m.doc() = "submodule for fast recursive calculation of probabilities in c++"; 

//    m.def("calculate_P_full", &calculate_P_full, "A function that calculates probability of a G(n,p) graph to have a largest connected component with i nodes.");
//}