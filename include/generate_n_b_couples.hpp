#ifndef GENERATE_N_B_COUPLES
#define GENERATE_N_B_COUPLES

// c++ libraries
// for std::vector
#include <vector>
// for std::pair
#include <utility>
// for std::string
#include <string>

// generate the list of couples (n,b)
std::vector<std::pair<int, int>> generate_list_of_all_n_b(int min_input_size, int max_input_size, int max_num_of_b_per_n, double to_multiply, int to_sum, int min_blocking_factor, int max_num_tests, int seed);

// prints to file the list of couples (n,b)
void print_list_to_file(std::vector<std::pair<int, int>> list_of_all_n_b, std::string filename);

// MACROS 

// DEFAULT VALUES:
// returns a random number between 1.300 and 1.600
#define OBTAIN_VAL_TO_MULTIPLY(rand_value) ((double) ((rand_value % 300) + 1300)) / ((double) 1000)
// outputs a random even number between 0 and 100
#define OBTAIN_VAL_TO_SUM(rand_value) ((rand_value % 101) * 2)

#endif //GENERATE_N_B_COUPLES
