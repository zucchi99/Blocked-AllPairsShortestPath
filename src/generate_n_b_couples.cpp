#include "../include/generate_n_b_couples.hpp"
#include "../include/macros.hpp"

// c library
// for srand() and rand()
#include <time.h>
#include <stdlib.h>

// c++ library for file stream
#include <fstream>

std::vector<std::pair<int, int>> generate_list_of_all_n_b(int min_input_size, int max_input_size, int max_num_of_b_per_n, double to_multiply, int to_sum, int min_blocking_factor, int max_num_tests, int seed) {

    // initialize seed
    srand(seed);

    std::vector<std::pair<int, int>> list_of_all_n_b;

    for (int n = min_input_size; n <= max_input_size; n = mmax(((int) (to_multiply * (double) n)) + to_sum, (n+1)) ) {

        // use mmax 5 different blocking factors
        int B_used[max_num_of_b_per_n];
        // initially all are -1
        for (int i = 0; i < max_num_of_b_per_n; i++) B_used[i] = -1;

        // index of the currently used B 
        int cur_B_idx = -1;

        // generate randomly B check if it is a divisor of n and not already used.
        // generate maximum max_num_tests random B (necessary to avoid non-termination, maybe n is prime)
        for (int tests = 0; tests < max_num_tests && cur_B_idx < max_num_of_b_per_n; tests++) {

            // range for b is between 0 and n/2
            int B = rand() % mmin(n/2, MAX_BLOCKING_FACTOR);
            // but if it is zero then use B=n
            B = (B == 0) ? n : B;

            // test if it is ok to be executed (b is a new divisor)
            bool is_ok = (n % B == 0) && (B <= MAX_BLOCKING_FACTOR) && (B >= min_blocking_factor); 
            for (int i = 0; (i <= cur_B_idx) && is_ok; i++) is_ok = (B != B_used[i]);

            if (is_ok) {
                B_used[++cur_B_idx] = B;
                list_of_all_n_b.push_back(std::make_pair(n, B));
                //printf("n: %d, b: %d\n", n, B);
            }
        }
    }

    return list_of_all_n_b;
}

void print_list_to_file(std::vector<std::pair<int, int>> list_of_all_n_b, std::string filename) {

    std::ofstream myfile;
    myfile.open(filename);
    myfile << "n,b\n";
        
    for (int i = 0; i < list_of_all_n_b.size(); i++) {

        int n = list_of_all_n_b[i].first;
        int b = list_of_all_n_b[i].second;
        myfile << n << "," << b << "\n";
    }
    myfile.close();
}

