
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

    std::string version = str_args[0];

    std::string exec_option;
    if (argc > 1) exec_option = str_args[1];

    if (argc == 1 || exec_option == "--help") {
        printf("Usage: %s <exec_option> [--version=<version>] [--analyzer=<nvprof|chrono> --input-file=<file>, --output-file=<file>, -n=<n>, -b=<b>, -t=<t> [-s=<s>]]:\n", argv[0]);
        printf(" where <exec_option>=test for statistical testing or <exec_option>=perf for performance analysis, <exec_option>=launch for basic execution\n");
        printf("If <exec_option>=perf and matrix must be randomly generated then specify n (matrix dimension), b (blocking factor), t (number of tests), [ s (seed), by default is random ]\n");
        printf("If <exec_option>=perf and matrix must be imported from csv then specify input-file (of matrix), n (matrix dimension), b (blocking factor), t (number of tests)\n");
        printf("If <exec_option>=launch and matrix must be randomly generated then specify input-file (of matrix), n (matrix dimension), b (blocking factor)\n");
        printf("If <exec_option>=launch and matrix must be imported from csv then specify input-file (of matrix), n (matrix dimension), b (blocking factor)\n");
        return 1;
    }

    // test no parameters to read
    if (exec_option == "test") {

        MultiSizeTestParameters my_params;
        my_params.f = f;
        my_params.g = &host_array_floyd_warshall_blocked;
        my_params.start_input_size = 30;
        my_params.end_input_size = 1000;
        my_params.to_multiply = RANDOM_CONSTANT;
        my_params.to_sum      = RANDOM_CONSTANT;
        my_params.min_blocking_factor = 2;

        multi_size_statistical_test(my_params);
        return 0;

    }
    
    //default values
    int *matrix;
    matrix = NULL;
    bool rand_matrix = true;
    std::string input_file = "csv/input_matrix.csv";
    std::string output_file = "csv/chrono_performances.csv";
    std::string analyzer = "chrono";
    int n = -1, b = -1, t = -1, s = -1;

    for (int i = 2; i < argc; i++) {

        std::string param = str_args[i];
        int index_of_equal = param.find('=');
        std::string param_name = param.substr(0, index_of_equal + 1);
        std::string param_val = param.substr(index_of_equal + 1);

        if(param[0] == '-' && param[2] == '=') {
            int val = std::stoi(param.substr(3));
            if(     param[1] == 'n') n = val; // mandatory: matrix size
            else if(param[1] == 'b') b = val; // mandatory: blocking factor size
            else if(param[1] == 't') t = val; // mandatory: number of tests to execute
            else if(param[1] == 's') s = val; // optional:  seed

        } else if(param_name == "--output-file=") {
            output_file = param_val;
            
        } else if(param_name == "--input-file=") {
            
            input_file = param_val;

            printf("input_file: %s\n", input_file.c_str());

            matrix = allocate_arr_matrix(n, n);

            // read matrix from csv
            int *realN; // (not used)

            read_arr_matrix(matrix, realN, input_file, ' ');

            print_arr_matrix(matrix, *realN, *realN);

            // matrix = read_csv(input_file);
            
        } else if(param_name == "--analyzer=") {
            if (param_val == "chrono" || param_val == "nvprof" ) {
                analyzer = param_val;
            }
              
        } else if(param_name == "--version=") {
            version = param_val;
            
        } else {
            printf("Parameter %s not recognised\n", param_name.c_str());
            return 4;
        }
    }
    
    rand_matrix = (matrix == NULL);

    bool is_perf   = (exec_option == "perf");
    bool is_launch = (exec_option == "launch");

    if (n <= 0) {
        printf("n must be specified and must be positive integers\n");
        return 5;

    } 
    if(b <= 0) {
        printf("b must be specified and must be positive integers\n");
        return 5;

    } 
    if ((b > n) || (n % b > 0)) {
        printf("b must be a divisor of n\n");
        return 6;
    }
    
    if (is_perf) {

        if (t <= 0) {
            printf("t must be specified and must be positive integers\n");
            return 5;
        }

        //if (s == -1) s = time(NULL);
        //printf("seed: %d\n", s);

        if (analyzer == "chrono") {
            do_chrono_performance_test(f, n, b, t, s, version, output_file, matrix, rand_matrix);
        } else {
            do_nvprof_performance_test(f, n, b, t, s, matrix, rand_matrix);
        }

    } else if (is_launch) {

        // just launch

        if (rand_matrix) {
            printf("missing matrix, with <exec_option>=launch an input csv matrix is needed\n");
            return 7;
        }

        printf("input_size: %d, blocking_factor: %d, number_of_executions: 1\n", n, b);

        // print input
        printf("input matrix:\n");
        print_arr_matrix(matrix, n, n);

        // launch f
        f(matrix, n, b);

        // print output
        printf("output matrix:\n");
        print_arr_matrix(matrix, n, n);

    } else {
        printf("<exec_option>=%s not recognised, try run: %s --help\n", argv[1], argv[0]);
        return 8;
    }

    return 0;
}