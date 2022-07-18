#include "../include/performance_test.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <cuda_profiler_api.h>

#include "../include/adj_matrix_utils.hpp"


void do_nvprof_performance_test(void (*floyd_warshall_arr_algorithm)(int * matrix, int n, int B), int input_size, int blocking_factor, int number_of_tests, int seed) {

    int* arr_matrix = (int *) malloc(sizeof(int *) * input_size * input_size);

    for (int i=0; i<number_of_tests; i++) {

        populate_arr_graph(arr_matrix, input_size, seed*(i+1));

        cudaProfilerStart();
        floyd_warshall_arr_algorithm(arr_matrix, input_size, blocking_factor);
        cudaProfilerStop();

        printf("Performed test number %d\n", i);
    }
}