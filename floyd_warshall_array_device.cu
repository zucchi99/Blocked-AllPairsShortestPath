
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>

#include <ctime>


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_profiler_api.h>

#include "include/adj_matrix_utils.hpp"
#include "include/host_floyd_warshall.hpp"
#include "include/cuda_errors_utils.cuh"
#include "include/performance_test.cuh"


void floyd_warshall_blocked_device_v1_0(int *matrix, int n, int B);
__global__ void execute_round_device(int *matrix, int n, int t, int row, int col, int B);


void temp_statistical_test(int n_tests, size_t input_size, int BLOCKING_FACTOR, bool stop_if_fail) {

    int n_wrong = 0;

    for (size_t i = 0; i < n_tests; i++)
    {
        //random seed
        int rand_seed = clock(); // (i+1)*clock(); //time(NULL);
        // srand(rand_seed);
        printf("%lu/%d)\tseed: %d", i, n_tests, rand_seed);

        //matrix initialization
        int *rand_matrix_1 = (int *) malloc(sizeof(int *) * input_size * input_size);
        int *rand_matrix_2 = (int *) malloc(sizeof(int *) * input_size * input_size);
        populate_arr_graph(rand_matrix_1, input_size, rand_seed);
        copy_arr_graph(rand_matrix_1, rand_matrix_2, input_size);
        // generate_arr_graph(input_size, rand_seed);

        //floyd_warshall execution
        arr_floyd_warshall(rand_matrix_1, input_size);

        //---------------------------------------------------------------

        //matrix initialization with same seed
        //  int *rand_matrix_2 = generate_arr_graph(input_size, rand_seed);
        
        //floyd_warshall_blocked execution (on device)
        floyd_warshall_blocked_device_v1_0(rand_matrix_2, input_size, BLOCKING_FACTOR);
        // arr_floyd_warshall_blocked(rand_matrix_2, input_size, BLOCKING_FACTOR);
        
        //---------------------------------------------------------------

        //compare matrixes output
        bool are_the_same = same_arr_matrix(rand_matrix_1, rand_matrix_2, input_size);

        if (!are_the_same) {

            n_wrong++;

            // //matrix print
            // printf("\ninput adjacency matrix %lux%lu:\n", input_size, input_size);
            // print_arr_matrix(rand_matrix_1, input_size, input_size);

            // //print floyd_warshall output
            // printf("output adjacency matrix classic %lux%lu:\n", input_size, input_size);
            // print_arr_matrix(rand_matrix_1, input_size, input_size);

            // //print floyd_warshall_blocked output
            // printf("output adjacency matrix blocked %lux%lu:\n", input_size, input_size);
            // print_arr_matrix(rand_matrix_2, input_size, input_size);
            // printf("Matrixes are equal? %s\n", bool_to_string(are_the_same));

            printf(" ERROR!\n"); 
            
            if (stop_if_fail) break;
        } else {
            printf("\tOK!\n");
        }

        free(rand_matrix_1);
        free(rand_matrix_2);
    }

    printf("%d errors detected\n\n", n_wrong);
}


int main() {

    //matrix size n*n
    size_t n = 128;

    //if no weights in graph:
    //int INF = (n * (n-1) / 2) + 1;

    int BLOCKING_FACTOR = 16;

    // temp_statistical_test(10000, n, BLOCKING_FACTOR, true);
    // do_nvprof_performance_test(&floyd_warshall_blocked_device_v1_0, n, BLOCKING_FACTOR, 100, clock());
    
    return 0;
}

__global__ void execute_round_device(int *matrix, int n, int t, int row, int col, int B) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    int i = tid/n;  // row
    int j = tid%n;  // col

    //foreach k: t*B <= t < t+B
    for (int k = t * B; k < (t+1) * B; k++) {

        int a, b;

        // check if thread correspond to one of the cells in current block
        if (i>=row * B && i<(row+1) * B && j>=col * B && j<(col+1) * B) {

            // WARNING: do NOT put the macro directly into 
            a = matrix[i*n + j];
            b = sum_if_not_infinite(matrix[i*n + k], matrix[k*n + j], INF); 
        }

        __syncthreads();


        if (i>=row * B && i<(row+1) * B && j>=col * B && j<(col+1) * B) {
            matrix[i*n + j] = min(a, b);
        }
        
        __syncthreads();

    }
}



void floyd_warshall_blocked_device_v1_0(int *matrix, int n, int B) {

    int *dev_rand_matrix;
    HANDLE_ERROR(cudaMalloc( (void**) &dev_rand_matrix, n * n* sizeof(int)));
    HANDLE_ERROR(cudaMemcpy(dev_rand_matrix, matrix, n*n*sizeof(int), cudaMemcpyHostToDevice));
    
    int num_rounds = n/B;

    int num_blocks = num_rounds*num_rounds;
    int thread_per_block = B*B; 
    

    for(int t = 0; t < num_rounds; t++) { 

        //arr_execute_round(int *matrix, int n, int t, int row, int col, int B)

        //phase 1: self-dependent block
        execute_round_device<<<num_blocks, thread_per_block>>>(dev_rand_matrix, n, t, t, t, B);

        //phase 2 blocks left
        for (int j = t-1; j >= 0; j--) {
            execute_round_device<<<num_blocks, thread_per_block>>>(dev_rand_matrix, n, t, t, j, B);
        }

        //phase 2 blocks above
        for (int i = t-1; i >= 0; i--) {
            execute_round_device<<<num_blocks, thread_per_block>>>(dev_rand_matrix, n, t, i, t, B);
        }

        //phase 2 blocks below
        for (int i = t+1; i < num_rounds; i++) {
            execute_round_device<<<num_blocks, thread_per_block>>>(dev_rand_matrix, n, t, i, t, B);
        }

        //phase 2 blocks right
        for (int j = t+1; j < num_rounds; j++) {
            execute_round_device<<<num_blocks, thread_per_block>>>(dev_rand_matrix, n, t, t, j, B);
        }
        
        //phase 2,3: remaining blocks
        //phase 3 blocks above and right
        for (int j = t+1; j < num_rounds; j++) {
            for (int i = t-1; i >= 0; i--) {
                execute_round_device<<<num_blocks, thread_per_block>>>(dev_rand_matrix, n, t, i, j, B);
            }
        }
        //phase 3 blocks above and left
        for (int j = t-1; j >= 0; j--) {
            for (int i = t-1; i >= 0; i--) {
                execute_round_device<<<num_blocks, thread_per_block>>>(dev_rand_matrix, n, t, i, j, B);
            }
        }
        //phase 3 blocks below and left
        for (int j = t-1; j >= 0; j--) {
            for (int i = t+1; i < num_rounds; i++) {
                execute_round_device<<<num_blocks, thread_per_block>>>(dev_rand_matrix, n, t, i, j, B);
            }
        }      
        //phase 3 blocks below and right
        for (int j = t+1; j < num_rounds; j++) {
            for (int i = t+1; i < num_rounds; i++) {
                execute_round_device<<<num_blocks, thread_per_block>>>(dev_rand_matrix, n, t, i, j, B);
            }
        }   
    }

    HANDLE_ERROR(cudaMemcpy(matrix, dev_rand_matrix, n*n*sizeof(int), cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaFree(dev_rand_matrix));
}