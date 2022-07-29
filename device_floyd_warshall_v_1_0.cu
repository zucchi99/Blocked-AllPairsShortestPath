
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>

#include <ctime>
#include <cassert>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_profiler_api.h>

#include "include/adj_matrix_utils.cuh"
#include "include/adj_matrix_utils.hpp"
#include "include/cuda_errors_utils.cuh"
#include "include/host_floyd_warshall.hpp"
#include "include/macros.hpp"
#include "include/statistical_test.hpp"

//main device code
void floyd_warshall_blocked_device_v_1_0(int *matrix, int n, int B);

//rounds code
__global__ void execute_round_device_v_1_0(int *matrix, int n, int t, int row, int col, int B);

int main() {

    // for (size_t n = 10; n < 200; n += 2) {

    //     int MAX_B = min(32, n);
    
    //     for (int BLOCKING_FACTOR = 1; BLOCKING_FACTOR < MAX_B; BLOCKING_FACTOR += 2) {

    //         if((n % BLOCKING_FACTOR) == 0) {
                
    //             printf("n: %ld, B: %d\n", n, BLOCKING_FACTOR);
    //             int n_err = do_arr_floyd_warshall_statistical_test(&floyd_warshall_blocked_device_v_1_0, n, BLOCKING_FACTOR, 1000, RANDOM_SEED, true, 4, true);
    //             // int n_err = do_arr_floyd_warshall_statistical_test(&arr_floyd_warshall_blocked, n, BLOCKING_FACTOR, 1000, RANDOM_SEED, true, 4, true);

    //             if (n_err>0) return;
    //         }
    //     }
    // }

    multi_size_statistical_test(&floyd_warshall_blocked_device_v_1_0, 128, 256, 8, 32, 100, RANDOM_SEED, false, false);

    // int n = 128;
    // int b = 16;
    // int n_tests = 1000;
    // // int seed = 2862999;
    // int seed = RANDOM_SEED;

    // do_arr_floyd_warshall_statistical_test(&floyd_warshall_blocked_device_v_1_0, n, b, n_tests, seed, false, 4, true);

    // 
    // do_nvprof_performance_test(&floyd_warshall_blocked_device_v_1_0, n, BLOCKING_FACTOR, 100, clock());
    

    // int *input_instance = (int *) malloc(sizeof(int *) * n * n);
    // int *test_instance_space = (int *) malloc(sizeof(int *) * n * n);
    // populate_arr_graph(input_instance, n, seed);
    // copy_arr_graph(input_instance, test_instance_space, n);
    // bool result = test_arr_floyd_warshall(&floyd_warshall_blocked_device_v_1_0, input_instance, test_instance_space, n, b);
    // printf("Corretto: %s\n", bool_to_string(result));

    return 0;
}

__global__ void execute_round_device_v_1_0(int *matrix, int n, int t, int row, int col, int B) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    int i = tid/n;  // row
    int j = tid%n;  // col

    //foreach k: t*B <= t < t+B
    for (int k = t * B; k < (t+1) * B; k++) {

        int a, b;
        bool run_this = ((i >= row*B) && (i < (row+1)*B) && (j >= col*B) && (j < (col+1)*B));

        // check if thread correspond to one of the cells in current block
        if (run_this) {

            // WARNING: do NOT put the macro directly into 
            a = matrix[i*n + j];
            b = sum_if_not_infinite(matrix[i*n + k], matrix[k*n + j], INF); 
        }

        __syncthreads();


        if (run_this) {
            matrix[i*n + j] = min(a, b);
        }
        
        __syncthreads();

    }
}


void floyd_warshall_blocked_device_v_1_0(int *matrix, int n, int B) {

    int *dev_rand_matrix;
    HANDLE_ERROR(cudaMalloc( (void**) &dev_rand_matrix, n * n* sizeof(int)));
    HANDLE_ERROR(cudaMemcpy(dev_rand_matrix, matrix, n*n*sizeof(int), cudaMemcpyHostToDevice));
    
    int num_rounds = n/B;

    int num_blocks = num_rounds*num_rounds;
    int thread_per_block = B*B; 
    

    for(int t = 0; t < num_rounds; t++) { 

        //arr_execute_round(int *matrix, int n, int t, int row, int col, int B)

        //phase 1: self-dependent block
        execute_round_device_v_1_0<<<num_blocks, thread_per_block>>>(dev_rand_matrix, n, t, t, t, B);
        HANDLE_ERROR(cudaDeviceSynchronize());

        //phase 2 blocks left
        for (int j = t-1; j >= 0; j--) {
            execute_round_device_v_1_0<<<num_blocks, thread_per_block>>>(dev_rand_matrix, n, t, t, j, B);
            // HANDLE_ERROR(cudaDeviceSynchronize());  
        }

        //phase 2 blocks above
        for (int i = t-1; i >= 0; i--) {
            execute_round_device_v_1_0<<<num_blocks, thread_per_block>>>(dev_rand_matrix, n, t, i, t, B);
            // HANDLE_ERROR(cudaDeviceSynchronize());  
        }

        //phase 2 blocks below
        for (int i = t+1; i < num_rounds; i++) {
            execute_round_device_v_1_0<<<num_blocks, thread_per_block>>>(dev_rand_matrix, n, t, i, t, B);
            // HANDLE_ERROR(cudaDeviceSynchronize());  
        }

        //phase 2 blocks right
        for (int j = t+1; j < num_rounds; j++) {
            execute_round_device_v_1_0<<<num_blocks, thread_per_block>>>(dev_rand_matrix, n, t, t, j, B);
            // HANDLE_ERROR(cudaDeviceSynchronize());  
        }

        HANDLE_ERROR(cudaDeviceSynchronize());
        
        //phase 3 blocks above and right
        for (int j = t+1; j < num_rounds; j++) {
            for (int i = t-1; i >= 0; i--) {
                execute_round_device_v_1_0<<<num_blocks, thread_per_block>>>(dev_rand_matrix, n, t, i, j, B);
                // HANDLE_ERROR(cudaDeviceSynchronize());  
            }
        }
        //phase 3 blocks above and left
        for (int j = t-1; j >= 0; j--) {
            for (int i = t-1; i >= 0; i--) {
                execute_round_device_v_1_0<<<num_blocks, thread_per_block>>>(dev_rand_matrix, n, t, i, j, B);
                // HANDLE_ERROR(cudaDeviceSynchronize());  
            }
        }
        //phase 3 blocks below and left
        for (int j = t-1; j >= 0; j--) {
            for (int i = t+1; i < num_rounds; i++) {
                execute_round_device_v_1_0<<<num_blocks, thread_per_block>>>(dev_rand_matrix, n, t, i, j, B);
                // HANDLE_ERROR(cudaDeviceSynchronize());  
            }
        }      
        //phase 3 blocks below and right
        for (int j = t+1; j < num_rounds; j++) {
            for (int i = t+1; i < num_rounds; i++) {
                execute_round_device_v_1_0<<<num_blocks, thread_per_block>>>(dev_rand_matrix, n, t, i, j, B);
                // HANDLE_ERROR(cudaDeviceSynchronize());  
            }
        }

        // HANDLE_ERROR(cudaDeviceSynchronize());   
    }

    HANDLE_ERROR(cudaDeviceSynchronize());  

    HANDLE_ERROR(cudaMemcpy(matrix, dev_rand_matrix, n*n*sizeof(int), cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaFree(dev_rand_matrix));
}

