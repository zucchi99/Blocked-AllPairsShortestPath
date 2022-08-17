#include "include/generate_n_b_couples.hpp"
#include "time.h"

int main() {

    int seed = time(NULL);
    srand(seed);

    // define parameters
    double costant_multiplier = ((double) ((rand() % 300) + 1300)) / ((double) 1000);
    int min_input_size = 30;
    int max_input_size = 150;
    int min_blocking_factor = 2;
    int max_num_of_b_per_n = 5;
    int max_num_tests = 50;

    // print parameters
    printf("costant_multiplier:\t%f\n", costant_multiplier);
    printf("seed:\t\t\t%d\n", seed);
    printf("min_input_size:\t\t%d\n", min_input_size);
    printf("max_input_size:\t\t%d\n", max_input_size);
    printf("min_blocking_factor:\t%d\n", min_blocking_factor);
    printf("max_num_of_b_per_n:\t%d\n", max_num_of_b_per_n);
    printf("max_num_tests:\t\t%d\n", max_num_tests);

    // generate list
    std::vector<std::pair<int, int>> list_of_all_n_b = generate_list_of_all_n_b(min_input_size, max_input_size, max_num_of_b_per_n, costant_multiplier, min_blocking_factor, max_num_tests, seed);

    // define output file name
    std::string out_filename = "csv/list_of_n_b.csv";

    // print list to file
    print_list_to_file(list_of_all_n_b, out_filename);

    printf("\noutput file:\t\t%s\n", out_filename.c_str());

    return 0;
    
}