
#include "../include/adj_matrix_utils.cuh"
#include "../include/adj_matrix_utils.hpp"
#include "../include/cuda_errors_utils.cuh"
#include "../include/host_floyd_warshall.hpp"
#include "../include/macros.hpp"
#include "../include/performance_test.cuh"
#include "../include/statistical_test.hpp"

// c libraries
// for printing
#include <stdio.h>
// to define a seed using current time with time(NULL)
#include <time.h>
// for pseudo-random generation with srand() for seed initialization and rand() for pseudo-random number
#include <stdlib.h>

// c++ libraries
// for argv handling as vector of string
#include <string>
#include <vector>

int handle_arguments_and_execute(int argc, char *argv[], void (*f) (int* arr_matrix, int n, int b)) {

    std::vector<std::string> str_args(argv, argv + argc);

    std::string exec_option;
    if (argc > 1) exec_option = str_args[1];

    if (argc == 1 || exec_option == "--help") {
        printf("Usage: %s <exec_option> [-n=<n>, -b=<b>, -t=<t>]:\n", argv[0]);
        printf(" where <exec_option>=test for statistical testing or <exec_option>=perf for nvprof profiling\n");
        printf("If <exec_option>=perf then specify n (matrix dimension), b (blocking factor), t (number of tests)\n");
        return 1;
    }

    if (exec_option == "test") {

        MultiSizeTestParameters my_params;
        my_params.f = f;
        my_params.g = &host_array_floyd_warshall_blocked;
        my_params.start_input_size = 30;
        my_params.end_input_size = 150;
        my_params.costant_multiplier = RANDOM_CONSTANT;
        my_params.min_blocking_factor = 2;

        print_multi_size_test_parameters(my_params);
        multi_size_statistical_test(my_params);
        

    } else if (exec_option == "perf") {
        
        int n = -1, b = -1, t = -1;
        if (str_args.size() < 5) {
            printf("Missing n,t,b parameters\n");
            return 2;
        }
        for (int i = 2; i < argc; i++) {
            if(str_args[i].size() < 4) {
                //mmin size for n,b,t parameters is 4 (ex. -n=5)
                printf("Uncorrect syntax parameter, use -<param>=<value>\n");
                return 3;
            }
            if(str_args[i][0] == '-' && str_args[i][2] == '=') {
                int val = std::stoi((str_args[i]).substr(3));
                if(     str_args[i][1] == 'n') n = val;
                else if(str_args[i][1] == 'b') b = val;
                else if(str_args[i][1] == 't') t = val;                        
                else {
                    printf("Parameter not recognised\n");
                    return 4;
                }
            } 
        }
        if (n <= 0 || b <= 0 || t <= 0) {
            printf("n, b, t must all be specified and must be positive integers\n");
            return 5;
        }
        if (b > n || (n % b > 0)) {
            printf("b must be a divisor of n\n");
            return 6;
        }
        int rand_seed = time(NULL);
        printf("rand_seed: %d\n", rand_seed);
        do_nvprof_performance_test(f, n, b, t, rand_seed);

    } else {
        printf("<exec_option>=%s not recognised, try run: %s --help\n", argv[1], argv[0]);
        return 7;
    }

    return 0;
}