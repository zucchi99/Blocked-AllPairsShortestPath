
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

//#include "include/adj_matrix_utils.cuh"
#include "include/adj_matrix_utils.hpp"
#include "include/host_floyd_warshall.hpp"
#include "include/cuda_errors_utils.cuh"
#include "include/performance_test.cuh"
#include "include/statistical_test.hpp"

#define MAX_BLOCK_SIZE 1024 // in realtà basta fare le proprerties della macchina

void floyd_warshall_blocked_device_v1_0(int *matrix, int n, int B);
__global__ void execute_round_device_v1_0(int *matrix, int n, int t, int row, int col, int B);

void floyd_warshall_blocked_device_v1_1(int *matrix, int n, int B);
__global__ void execute_round_device_v1_1(int *matrix, int n, int t, int row, int col, int B);

void floyd_warshall_blocked_device_v_pitch(int *matrix, int n, int B);

__device__ void print_matrix_device(int *matrix, int m, int n);


int main() {

    // for (size_t n = 10; n < 200; n += 2) {

    //     int MAX_B = min(32, n);
    
    //     for (int BLOCKING_FACTOR = 1; BLOCKING_FACTOR < MAX_B; BLOCKING_FACTOR += 2) {

    //         if((n % BLOCKING_FACTOR) == 0) {
                
    //             printf("n: %ld, B: %d\n", n, BLOCKING_FACTOR);
    //             int n_err = do_arr_floyd_warshall_statistical_test(&floyd_warshall_blocked_device_v1_0, n, BLOCKING_FACTOR, 1000, RANDOM_SEED, true, 4, true);
    //             // int n_err = do_arr_floyd_warshall_statistical_test(&arr_floyd_warshall_blocked, n, BLOCKING_FACTOR, 1000, RANDOM_SEED, true, 4, true);

    //             if (n_err>0) return;
    //         }
    //     }
    // }

    
    size_t n = 6;
    int BLOCKING_FACTOR = 2;
    printf("n: %ld, B: %d\n", n, BLOCKING_FACTOR);
    int n_err = do_arr_floyd_warshall_statistical_test(&floyd_warshall_blocked_device_v_pitch, n, BLOCKING_FACTOR, 1, RANDOM_SEED, true, 4, true);


    //multi_size_statistical_test(&floyd_warshall_blocked_device_v_pitch, 16, 512, 8, 32, 100, RANDOM_SEED, false, false);

    // int n = 128;
    // int b = 16;
    // int n_tests = 1000;
    // // int seed = 2862999;
    // int seed = RANDOM_SEED;

    // do_arr_floyd_warshall_statistical_test(&floyd_warshall_blocked_device_v1_1, n, b, n_tests, seed, false, 4, true);

    // 
    // do_nvprof_performance_test(&floyd_warshall_blocked_device_v1_0, n, BLOCKING_FACTOR, 100, clock());
    

    // int *input_instance = (int *) malloc(sizeof(int *) * n * n);
    // int *test_instance_space = (int *) malloc(sizeof(int *) * n * n);
    // populate_arr_graph(input_instance, n, seed);
    // copy_arr_graph(input_instance, test_instance_space, n);
    // bool result = test_arr_floyd_warshall(&floyd_warshall_blocked_device_v1_0, input_instance, test_instance_space, n, b);
    // printf("Corretto: %s\n", bool_to_string(result));

    return 0;
}


void floyd_warshall_blocked_device_v_pitch(int *matrix, int n, int B) {
    
    assert(n%B == 0);                       // B must divide n
    assert(B*B<=MAX_BLOCK_SIZE);            // B*B cannot exceed max block size

    int *dev_rand_matrix;
    size_t pitch;                          //size in bytes of memory allocated to guarantee alignment
    size_t width = n * sizeof(int);
    size_t height = n;

    //cudaMallocPitch(&devPtr, &devPitch, N_cols * sizeof(type), N_rows);

    HANDLE_ERROR(cudaMallocPitch( (void**) &dev_rand_matrix, &pitch, width, height));
    HANDLE_ERROR(cudaMemcpy(dev_rand_matrix, matrix, n * n * sizeof(int), cudaMemcpyHostToDevice));

    int num_rounds = n/B;
     
    for(int t = 0; t < num_rounds; t++) { 

        //arr_execute_round(int *matrix, int n, int t, int row, int col, int B)

        //phase 1: self-dependent block
        dim3 num_blocks_phase_1(1, 1);
        dim3 threads_per_block_phase_1(B, B);

        execute_round_device_v1_1<<<num_blocks_phase_1, threads_per_block_phase_1>>>(dev_rand_matrix, n, t, t, t, B);
        HANDLE_ERROR(cudaDeviceSynchronize());

        //phase 2 blocks left
        for (int j = t-1; j >= 0; j--) {
            execute_round_device_v1_1<<<num_blocks_phase_1, threads_per_block_phase_1>>>(dev_rand_matrix, n, t, t, j, B);
            // HANDLE_ERROR(cudaDeviceSynchronize());  
        }

        //phase 2 blocks above
        for (int i = t-1; i >= 0; i--) {
            execute_round_device_v1_1<<<num_blocks_phase_1, threads_per_block_phase_1>>>(dev_rand_matrix, n, t, i, t, B);
            // HANDLE_ERROR(cudaDeviceSynchronize());  
        }

        //phase 2 blocks below
        for (int i = t+1; i < num_rounds; i++) {
            execute_round_device_v1_1<<<num_blocks_phase_1, threads_per_block_phase_1>>>(dev_rand_matrix, n, t, i, t, B);
            // HANDLE_ERROR(cudaDeviceSynchronize());  
        }

        //phase 2 blocks right
        for (int j = t+1; j < num_rounds; j++) {
            execute_round_device_v1_1<<<num_blocks_phase_1, threads_per_block_phase_1>>>(dev_rand_matrix, n, t, t, j, B);
            // HANDLE_ERROR(cudaDeviceSynchronize());  
        }

        HANDLE_ERROR(cudaDeviceSynchronize());
        
        //phase 3 blocks above and right
        for (int j = t+1; j < num_rounds; j++) {
            for (int i = t-1; i >= 0; i--) {
                execute_round_device_v1_1<<<num_blocks_phase_1, threads_per_block_phase_1>>>(dev_rand_matrix, n, t, i, j, B);
                // HANDLE_ERROR(cudaDeviceSynchronize());  
            }
        }
        //phase 3 blocks above and left
        for (int j = t-1; j >= 0; j--) {
            for (int i = t-1; i >= 0; i--) {
                execute_round_device_v1_1<<<num_blocks_phase_1, threads_per_block_phase_1>>>(dev_rand_matrix, n, t, i, j, B);
                // HANDLE_ERROR(cudaDeviceSynchronize());  
            }
        }
        //phase 3 blocks below and left
        for (int j = t-1; j >= 0; j--) {
            for (int i = t+1; i < num_rounds; i++) {
                execute_round_device_v1_1<<<num_blocks_phase_1, threads_per_block_phase_1>>>(dev_rand_matrix, n, t, i, j, B);
                // HANDLE_ERROR(cudaDeviceSynchronize());  
            }
        }      
        //phase 3 blocks below and right
        for (int j = t+1; j < num_rounds; j++) {
            for (int i = t+1; i < num_rounds; i++) {
                execute_round_device_v1_1<<<num_blocks_phase_1, threads_per_block_phase_1>>>(dev_rand_matrix, n, t, i, j, B);
                // HANDLE_ERROR(cudaDeviceSynchronize());  
            }
        }

        HANDLE_ERROR(cudaDeviceSynchronize());  
    }
}

__global__ void execute_round_device_v1_1(int *matrix, int n, int t, int row, int col, int B) {

    int tid_x = threadIdx.x + blockIdx.x * blockDim.x;
    int tid_y = threadIdx.y + blockIdx.y * blockDim.y;

    int i = tid_x + row * B;  // row
    int j = tid_y + col * B;  // col

    printf(
        "tid_x:%d, tid_y:%d, i:%d, j:%d, threadIdx.x:%d, blockIdx.x:%d, blockDim.x:%d, threadIdx.y:%d, blockIdx.y:%d, blockDim.y:%d\n",
        tid_x, tid_y, i, j, threadIdx.x, blockIdx.x, blockDim.x, threadIdx.y, blockIdx.y, blockDim.y
    );

    //foreach k: t*B <= t < t+B
    for (int k = t * B; k < (t+1) * B; k++) {

        bool run_this = ((i >= row*B) && (i < (row+1)*B) && (j >= col*B) && (j < (col+1)*B));

        // check if thread correspond to one of the cells in current block
        if (run_this) {

            int ik = matrix[i*n + k];
            int kj = matrix[k*n + j];
            int ij_bef = matrix[i*n + j];

            int using_k_path = sum_if_not_infinite(matrix[i*n + k], matrix[k*n + j], INF); 

            if (using_k_path < matrix[i*n + j]) {
                matrix[i*n + j] = using_k_path;
            }

            int ij_aft = matrix[i*n + j];

            
            printf("i:%d, j:%d, k:%d, ik:%02d, kj:%02d, ij_bef:%02d, ij_aft:%02d\n", i, j, k, (min(ik, 99)), (min(kj, 99)), (min(ij_bef, 99)), (min(ij_aft, 99)));
        }
        
        __syncthreads();

        

        if(i%2 == 0 && j%2 == 0) {
            printf("k:%d\n",k);
            print_matrix_device(matrix, n, n);
        }

        
    }
}

__device__ void print_matrix_device(int *matrix, int m, int n) {
    printf("[\n");
    for (int i = 0; i < m; i++) {
        printf("  ");
        for (int j = 0; j < n; j++) {
            int val = matrix[i*n + j];
            if (val < INF)
                printf("%02d", val);
            else 
                printf("--");
            if (j < n-1) printf(", ");
        }
        printf("\n");
    }
    printf("]\n");
}