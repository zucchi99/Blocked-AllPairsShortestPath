#ifndef GENERATE_N_B_COUPLES
#define GENERATE_N_B_COUPLES

#undef mmax
#undef mmin

// c++ libraries
// for std::vector
#include <vector>
// for std::pair
#include <utility>
// for std::string
#include <string>

std::vector<std::pair<int, int>> generate_list_of_all_n_b(int min_input_size, int max_input_size, int max_num_of_b_per_n, double costant_multiplier, int min_blocking_factor, int max_num_tests, int seed);
void print_list_to_file(std::vector<std::pair<int, int>> list_of_all_n_b, std::string filename);

#endif //GENERATE_N_B_COUPLES
