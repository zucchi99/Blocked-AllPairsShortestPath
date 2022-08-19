#include "include/generate_n_b_couples.hpp"
#include "include/macros.hpp"

#include "time.h"

int main(int argc, char** argv) {

    int seed = time(NULL);
    srand(seed);

    // define parameters
    double to_multiply = 1;
    int to_sum = 80;
    int min_input_size = 80;
    int max_input_size = 480;

    //if blocking factors are calculated randomly
    int min_blocking_factor = 2;
    int max_num_of_b_per_n = 5;
    int max_num_tests = 50;
    //if blocking factors are given from argv
    int* B;
    if (argc > 1) {
        B = (int *) malloc(sizeof(int) * (argc-1));
    }

    std::vector<std::pair<int, int>> list_of_all_n_b;

    // print parameters
    printf("to_multiply:\t\t%f\n", to_multiply);
    printf("to_sum:\t\t\t%d\n", to_sum);
    printf("seed:\t\t\t%d\n", seed);
    printf("min_input_size:\t\t%d\n", min_input_size);
    printf("max_input_size:\t\t%d\n", max_input_size);

    if (argc == 1) {
        // generate blocking factors randomly
        printf("min_blocking_factor:\t%d\n", min_blocking_factor);
        printf("max_num_of_b_per_n:\t%d\n", max_num_of_b_per_n);
        printf("max_num_tests:\t\t%d\n", max_num_tests);
        
        // generate list randomly
        list_of_all_n_b = generate_list_of_all_n_b(min_input_size, max_input_size, max_num_of_b_per_n, to_multiply, to_sum, min_blocking_factor, max_num_tests, seed);

    } else {
        // read blocking factors from argv
        for (int i = 1; i < argc; i++) {
            B[i-1] = atoi(argv[i]);
        }
        for (int n = min_input_size; n <= max_input_size; n = mmax(((int) (to_multiply * (double) n)) + to_sum, (n+1)) ) {
            for (int i = 0; i < argc-1; i++) {
                if (n % B[i] == 0) {
                    list_of_all_n_b.push_back(std::make_pair(n, B[i]));
                }                
            }
        }
    }


    // define output file name
    std::string out_filename = "csv/list_of_n_b.csv";

    // print list to file
    print_list_to_file(list_of_all_n_b, out_filename);

    // print file name
    printf("\noutput file:\t\t%s\n", out_filename.c_str());

    return 0;
    
}