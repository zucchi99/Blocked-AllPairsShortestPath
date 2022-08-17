
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>

#include <ctime>
#include <cassert>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "include/adj_matrix_utils.cuh"
#include "include/adj_matrix_utils.hpp"
#include "include/cuda_errors_utils.cuh"
#include "include/host_floyd_warshall.hpp"
#include "include/macros.hpp"
#include "include/performance_test.cuh"
#include "include/statistical_test.hpp"

//main device code
void floyd_warshall_blocked_device_v_pitch(int *matrix, int n, int B);

//rounds code
__global__ void execute_round_device_v_pitch(int *matrix, int n, int t, int row, int col, int B, size_t pitch);

int main() {

    // for (size_t n = 10; n < 200; n += 2) {

    //     int MAX_B = mmin(32, n);
    
    //     for (int BLOCKING_FACTOR = 1; BLOCKING_FACTOR < MAX_B; BLOCKING_FACTOR += 2) {

    //         if((n % BLOCKING_FACTOR) == 0) {
                
    //             printf("n: %ld, B: %d\n", n, BLOCKING_FACTOR);
    //             int n_err = do_arr_floyd_warshall_statistical_test(&floyd_warshall_blocked_device_v1_0, n, BLOCKING_FACTOR, 1000, RANDOM_SEED, true, 4, true);
    //             // int n_err = do_arr_floyd_warshall_statistical_test(&arr_floyd_warshall_blocked, n, BLOCKING_FACTOR, 1000, RANDOM_SEED, true, 4, true);

    //             if (n_err>0) return;
    //         }
    //     }
    // }

    //multi_size_statistical_test(&floyd_warshall_blocked_device_v_pitch, 128, 256, 8, 32, 100, RANDOM_SEED, false, false);

    int n = 256;
    int B = 32;
    //int *test_instance = allocate_arr_matrix(n, n);
    //int *input_instance = allocate_arr_matrix(n, n);
    int rand_seed = time(NULL);
    printf("rand_seed: %d\n", rand_seed);
    //populate_arr_adj_matrix(input_instance, n, rand_seed, false);
    do_nvprof_performance_test(&floyd_warshall_blocked_device_v_pitch, n, B, 10, rand_seed);
    //printf("input matrix:\n");
    //print_arr_matrix(input_instance, n, n);
    //printf("\n\n");
    //floyd_warshall_blocked_device_v_pitch(input_instance, n, B);
    //bool result = test_arr_floyd_warshall(&floyd_warshall_blocked_device_v_pitch, input_instance, test_instance, n, BLOCKING_FACTOR);
    //printf("Corretto: %s\n", bool_to_string(result));

    return 0;
}


void floyd_warshall_blocked_device_v_pitch(int *matrix, int n, int B) {
    
    assert(n%B == 0);                       // B must divide n
    assert(B*B<=MAX_BLOCK_SIZE);            // B*B cannot exceed mmax block size

    int *dev_rand_matrix;
    size_t pitch;                          //size in bytes of memory allocated to guarantee alignment
    size_t width = n * sizeof(int);
    size_t height = n;

    //cudaMallocPitch(&devPtr, &devPitch, N_cols * sizeof(type), N_rows);

    HANDLE_ERROR(cudaMallocPitch( (void**) &dev_rand_matrix, &pitch, width, height));
    //HANDLE_ERROR(cudaMemcpy(dev_rand_matrix, matrix, n * n * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy2D(dev_rand_matrix, pitch, matrix, width, width, height, cudaMemcpyHostToDevice));


    int num_rounds = n/B;
     
    for(int t = 0; t < num_rounds; t++) { 

        //arr_execute_round(int *matrix, int n, int t, int row, int col, int B)

        //phase 1: self-dependent block
        dim3 num_blocks_phase_1(1, 1);
        dim3 threads_per_block_phase_1(B, B);

        // printf("start self dependent, t:%d, row:%d, col:%d\n",t,t,t);
        execute_round_device_v_pitch<<<num_blocks_phase_1, threads_per_block_phase_1>>>(dev_rand_matrix, n, t, t, t, B, pitch);
        HANDLE_ERROR(cudaDeviceSynchronize());

        //phase 2 blocks left
        for (int j = t-1; j >= 0; j--) {
            // printf("start phase 2 blocks left, t:%d, row:%d, col:%d\n",t,t,j);
            execute_round_device_v_pitch<<<num_blocks_phase_1, threads_per_block_phase_1>>>(dev_rand_matrix, n, t, t, j, B, pitch);
            // HANDLE_ERROR(cudaDeviceSynchronize());  
        }

        //phase 2 blocks above
        for (int i = t-1; i >= 0; i--) {
            // printf("start phase 2 blocks above, t:%d, row:%d, col:%d\n",t,i,t);
            execute_round_device_v_pitch<<<num_blocks_phase_1, threads_per_block_phase_1>>>(dev_rand_matrix, n, t, i, t, B, pitch);
            // HANDLE_ERROR(cudaDeviceSynchronize());  
        }

        //phase 2 blocks below
        for (int i = t+1; i < num_rounds; i++) {
            // printf("start phase 2 blocks below, t:%d, row:%d, col:%d\n",t,i,t);
            execute_round_device_v_pitch<<<num_blocks_phase_1, threads_per_block_phase_1>>>(dev_rand_matrix, n, t, i, t, B, pitch);
            // HANDLE_ERROR(cudaDeviceSynchronize());  
        }

        //phase 2 blocks right
        for (int j = t+1; j < num_rounds; j++) {
            // printf("start phase 2 blocks right, t:%d, row:%d, col:%d\n",t,t,j);
            execute_round_device_v_pitch<<<num_blocks_phase_1, threads_per_block_phase_1>>>(dev_rand_matrix, n, t, t, j, B, pitch);
            // HANDLE_ERROR(cudaDeviceSynchronize());  
        }

        HANDLE_ERROR(cudaDeviceSynchronize());
        
        //phase 3 blocks above and right
        for (int j = t+1; j < num_rounds; j++) {
            for (int i = t-1; i >= 0; i--) {
                // printf("start phase 3 blocks above and right, t:%d, row:%d, col:%d\n",t,i,j);
                execute_round_device_v_pitch<<<num_blocks_phase_1, threads_per_block_phase_1>>>(dev_rand_matrix, n, t, i, j, B, pitch);
                // HANDLE_ERROR(cudaDeviceSynchronize());  
            }
        }
        //phase 3 blocks above and left
        for (int j = t-1; j >= 0; j--) {
            for (int i = t-1; i >= 0; i--) {
                // printf("start phase 3 blocks above and left, t:%d, row:%d, col:%d\n",t,i,j);
                execute_round_device_v_pitch<<<num_blocks_phase_1, threads_per_block_phase_1>>>(dev_rand_matrix, n, t, i, j, B, pitch);
                // HANDLE_ERROR(cudaDeviceSynchronize());  
            }
        }
        //phase 3 blocks below and left
        for (int j = t-1; j >= 0; j--) {
            for (int i = t+1; i < num_rounds; i++) {
                // printf("start phase 3 blocks below and left, t:%d, row:%d, col:%d\n",t,i,j);
                execute_round_device_v_pitch<<<num_blocks_phase_1, threads_per_block_phase_1>>>(dev_rand_matrix, n, t, i, j, B, pitch);
                // HANDLE_ERROR(cudaDeviceSynchronize());  
            }
        }      
        //phase 3 blocks below and right
        for (int j = t+1; j < num_rounds; j++) {
            for (int i = t+1; i < num_rounds; i++) {
                // printf("start phase 3 blocks below and right, t:%d, row:%d, col:%d\n",t,i,j);
                execute_round_device_v_pitch<<<num_blocks_phase_1, threads_per_block_phase_1>>>(dev_rand_matrix, n, t, i, j, B, pitch);
                // HANDLE_ERROR(cudaDeviceSynchronize());  
            }
        }

        HANDLE_ERROR(cudaDeviceSynchronize());  
    }

    // HANDLE_ERROR(cudaMemcpy(matrix, dev_rand_matrix, n*n*sizeof(int), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy2D(matrix, width, dev_rand_matrix, pitch, width, height, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaFree(dev_rand_matrix));
}

__global__ void execute_round_device_v_pitch(int *matrix, int n, int t, int row, int col, int B, size_t pitch) {

    int tid_x = threadIdx.x + blockIdx.x * blockDim.x;
    int tid_y = threadIdx.y + blockIdx.y * blockDim.y;
    
    int i = tid_x + row * B;  // row
    int j = tid_y + col * B;  // col
    
    int *cell_i_j = pitched_pointer(matrix, i, j, pitch); //(int *)((char*) matrix + i * pitch) + j;
    //int cell_i_j_bef = *cell_i_j;

    /*
    printf(
        "tid_x:%d, tid_y:%d, i:%d, j:%d, threadIdx.x:%d, blockIdx.x:%d, blockDim.x:%d, threadIdx.y:%d, blockIdx.y:%d, blockDim.y:%d\n",
        tid_x, tid_y, i, j, threadIdx.x, blockIdx.x, blockDim.x, threadIdx.y, blockIdx.y, blockDim.y
    );
    */

    //foreach k: t*B <= t < t+B
    for (int k = t * B; k < (t+1) * B; k++) {

        int* cell_k_j = pitched_pointer(matrix, k, j, pitch); //(int *)((char*) matrix + k * pitch) + j;
        int* cell_i_k = pitched_pointer(matrix, i, k, pitch); //(int *)((char*) matrix + i * pitch) + k;

        int using_k_path = sum_if_not_infinite(*cell_i_k, *cell_k_j, INF); 

        if (using_k_path < *cell_i_j) {
            *cell_i_j = using_k_path;
        }

        //printf("i:%d, j:%d, k:%d, max_k:%d, ik:%02d, kj:%02d, ij_bef:%02d, ij_aft:%02d\n", i, j, k, (t+1)*B, (mmin(*cell_i_k, 99)), (mmin(*cell_k_j, 99)), (mmin(cell_i_j_bef, 99)), (mmin(*cell_i_j, 99)));
        // if (tid_x==0 && tid_y==0) printf("i:%d, j:%d, k:%d, max_k:%d\n", i, j, k, (t+1)*B);      
        
        __syncthreads();


        /*
        if((i % 2 == 0) && (j % 2 == 0)) {
            printf("k:%d\n",k);
            print_matrix_device(matrix, n, n, pitch);
            printf("\n");
        }
        */

    }
    
}
