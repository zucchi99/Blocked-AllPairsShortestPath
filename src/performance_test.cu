
#include <stdio.h>
#include <stdlib.h>
#include <string>

#include "../include/adj_matrix_utils.hpp"
#include "../include/math.hpp"
#include "../include/performance_test.cuh"

// for nvidia profiler
#include <cuda_profiler_api.h>

using namespace std;
using namespace chrono;

void do_nvprof_performance_test(void (*floyd_warshall_arr_algorithm)(int* matrix, int n, int B), int input_size, int blocking_factor, int number_of_tests, int seed, int *in_matrix, bool rand_matrix) {

    int *arr_matrix = allocate_arr_matrix(input_size, input_size);

    for (int i=0; i < number_of_tests; i++) {

        if (rand_matrix) {
            //random population
            populate_arr_adj_matrix(arr_matrix, input_size, seed*(i+1), false);
        } else {
            //copy from input data
            copy_arr_matrix(arr_matrix, in_matrix, input_size, input_size);
        }

        cudaProfilerStart();
        floyd_warshall_arr_algorithm(arr_matrix, input_size, blocking_factor);
        cudaProfilerStop();

    }

    free(arr_matrix);
}

void do_chrono_performance_test(void (*floyd_warshall_arr_algorithm)(int *matrix, int n, int B), int input_size, int blocking_factor, int number_of_tests, int seed, string version, string output_file, int *in_matrix, bool rand_matrix) {
    
    // allocate matrix
    int* arr_matrix = allocate_arr_matrix(input_size, input_size);

    // calculate time needed to populate matrix
    duration<double> time_init = initialization_time(input_size, 200, seed);

    // obtain a vector of 20 time_exec values
    int mse_repetitions = 20;
    vector<double> time_exec_vec (mse_repetitions);
    double time_exec;

    // after execution check if measured_error is greater than relative_error
    // (the first time measured_error = -1 to be sure to execute while loop)
    
    steady_clock::time_point start, end;
    
    // repeat many times to obtain a vector of times and calculate mse
    for (int i = 0; i < mse_repetitions; i++) {
        
        start = steady_clock::now();
        // execute many times to be sure time_exec is big enough to respect error
        for (int j = 0; j < number_of_tests; j++) {

            if (rand_matrix) {
                //random population
                populate_arr_adj_matrix(arr_matrix, input_size, seed*(i+1), false);
            } else {
                //copy from input data
                copy_arr_matrix(arr_matrix, in_matrix, input_size, input_size);
            }
        
            floyd_warshall_arr_algorithm(arr_matrix, input_size, blocking_factor);
        }
        end = steady_clock::now();

        // store i-th time_exec
        duration<double, std::milli> time_diff_double = ((end - start - time_init) / number_of_tests);    
        time_exec_vec[i] = (time_diff_double).count();
    
        // obtain time_exec mean
        time_exec = mean(time_exec_vec);
    }
    
    double mse = mean_squared_error(time_exec_vec);
    double mse_perc = 100 * mse / mean(time_exec_vec);

    FILE *fp;
    fp = fopen(output_file.c_str(), "a");
    if (fp == NULL) {
        printf("failed opening file!\n");
    } else {
        fprintf(fp, "%s,%d,%d,%d,%d,%f,%f,%f%%\n", version.c_str(), seed, input_size, blocking_factor, number_of_tests, time_exec, mse, mse_perc);
        fclose(fp);
    }
    printf("input_size: %d, blocking_factor: %d, number_of_tests: %d, seed: %d\n", input_size, blocking_factor, number_of_tests, seed);
    printf("time_exec: %fms, mse: %fms, mse_perc: %f%%\n", time_exec, mse, mse_perc);

    free(arr_matrix);

}

// measure medium time needed to allocate a n*n matrix vector randomly, repetitions times
duration<double> initialization_time(int input_size, int repetitions, int seed, int *in_matrix, bool rand_matrix) {
    steady_clock::time_point start, end;
    int* arr_matrix = allocate_arr_matrix(input_size, input_size);
    start = steady_clock::now();
    for (int i = 0; i < repetitions; i++) {
        if (rand_matrix) {
            //random population
            populate_arr_adj_matrix(arr_matrix, input_size, seed*(i+1), false);
        } else {
            //copy from input data
            copy_arr_matrix(arr_matrix, in_matrix, input_size, input_size);
        }
    }
    end = steady_clock::now();
    return (duration<double>)((end - start) / repetitions);
}