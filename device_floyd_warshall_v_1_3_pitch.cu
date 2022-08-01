#include <stdlib.h>
#include <stdio.h>
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
void floyd_warshall_blocked_device_v_1_3_pitch(int *matrix, int n, int B);

//rounds code
__global__ void execute_round_device_v_1_2_phase_1(int *matrix, int n, int t, int B, size_t pitch);
__global__ void execute_round_device_v_1_3_phase_2(int *matrix, int n, int t, int B, size_t pitch);
__global__ void execute_round_device_v_1_3_phase_3(int *matrix, int n, int t, int B, size_t pitch);

int main() {

    //multi_size_statistical_test(&floyd_warshall_blocked_device_v_1_3, 8, 256, 8, 32, 1000, RANDOM_SEED, false, true);
    
    int n = 256;
    int B = 32;
    //int *test_instance = allocate_arr_matrix(n, n);
    //int *input_instance = allocate_arr_matrix(n, n);
    int rand_seed = time(NULL);
    printf("rand_seed: %d\n", rand_seed);
    //populate_arr_adj_matrix(input_instance, n, rand_seed, false);
    do_nvprof_performance_test(&floyd_warshall_blocked_device_v_1_3_pitch, n, B, 10, rand_seed);

    //single test
    /*
    size_t n = 6;
    int BLOCKING_FACTOR = 2;
    printf("n: %ld, B: %d\n", n, BLOCKING_FACTOR);
    int n_err = do_arr_floyd_warshall_statistical_test(&floyd_warshall_blocked_device_v_1_3, n, BLOCKING_FACTOR, 1, RANDOM_SEED, true, 4, true);
    printf("n_err:%d\n", n_err);
    */

    return 0;
}

void floyd_warshall_blocked_device_v_1_3_pitch(int *matrix, int n, int B) {

    assert(n%B == 0);                       // B must divide n
    assert(B*B<=MAX_BLOCK_SIZE);            // B*B cannot exceed max block size

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

        execute_round_device_v_1_2_phase_1<<<num_blocks_phase_1, threads_per_block_phase_1>>>(dev_rand_matrix, n, t, B, pitch);
        HANDLE_ERROR(cudaDeviceSynchronize());

        // phase 2: all blocks that share a row or a column with the self dependent, so
        //  -   all blocks just above or under t
        //  -   all block at left and at right of t

        dim3 num_blocks_phase_2(2, num_rounds-1);  

        execute_round_device_v_1_3_phase_2<<<num_blocks_phase_2, threads_per_block_phase_1>>>(dev_rand_matrix, n, t, B, pitch);
        HANDLE_ERROR(cudaDeviceSynchronize());

        // phase 3: all the remaining blocks, so all the blocks that don't share a row or a col with t

        dim3 num_blocks_phase_3(num_rounds-1, num_rounds-1); 

        execute_round_device_v_1_3_phase_3<<<num_blocks_phase_3, threads_per_block_phase_1>>>(dev_rand_matrix, n, t, B, pitch);
        HANDLE_ERROR(cudaDeviceSynchronize()); 
    }

    // HANDLE_ERROR(cudaMemcpy(matrix, dev_rand_matrix, n*n*sizeof(int), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy2D(matrix, width, dev_rand_matrix, pitch, width, height, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaFree(dev_rand_matrix));
}


__global__ void execute_round_device_v_1_2_phase_1(int *matrix, int n, int t, int B, size_t pitch) {

    // Launched block and correspondent position in the matrix

    //  t

    //  .   .   .   .   .   . 
    //  .   .   .   .   .   . 
    //  .   .   .   .   .   . 
    //  .   .   .   t   .   .
    //  .   .   .   .   .   . 
    //  .   .   .   .   .   . 

    int tid_x = threadIdx.x + blockIdx.x * blockDim.x;
    int tid_y = threadIdx.y + blockIdx.y * blockDim.y;

    int i = tid_x + t * B;  // row
    int j = tid_y + t * B;  // col

    int *cell_i_j = pitched_pointer(matrix, i, j, pitch);

    //foreach k: t*B <= t < t+B
    for (int k = BLOCK_START(t,B); k < BLOCK_END(t,B); k++) {

        int* cell_k_j = pitched_pointer(matrix, k, j, pitch); //(int *)((char*) matrix + k * pitch) + j;
        int* cell_i_k = pitched_pointer(matrix, i, k, pitch); //(int *)((char*) matrix + i * pitch) + k;

        int using_k_path = sum_if_not_infinite(*cell_i_k, *cell_k_j, INF); 

        if (using_k_path < *cell_i_j) {
            *cell_i_j = using_k_path;
        }
        
        __syncthreads();
    }
}

__global__ void execute_round_device_v_1_3_phase_2(int *matrix, int n, int t, int B, size_t pitch) {

    // Launched blocks and correspondent position in the matrix
    //  -   blockIdx.x says if I am iterating row or cols, 
    //  -   blockIdx.y says something about which row or col)
    //  -   threadIdx.x and threadIdx.y are relative position of cell in block

    //  L1  L2  L3  R1  R2
    //  U1  U2  U3  D1  D2

    //  .   .   .   U1  .   .
    //  .   .   .   U2  .   .
    //  .   .   .   U3  .   .
    //  L1  L2  L3  -   R1  R2
    //  .   .   .   D1  .   .
    //  .   .   .   D2  .   .

    int i, j;

    if (blockIdx.x == 0) {  

        // it's a row ...
        i = BLOCK_START(t, B) + threadIdx.x;

        if (blockIdx.y < t) {

            // ... and it's the left one
            j = BLOCK_START(blockIdx.y, B) + threadIdx.y;

        } else {
            
            // ... and it's the right one
            j = BLOCK_START(blockIdx.y, B) + B + threadIdx.y;
        }
    } else {

        // it's a column ...
        j = BLOCK_START(t, B) + threadIdx.y;

        if (blockIdx.y < t) {

            // ... and it's the up one
            i = BLOCK_START(blockIdx.y, B) + threadIdx.x;

        } else {

            // ... and it's the down one
            i = BLOCK_START(blockIdx.y, B) + B + threadIdx.x;
        }
    }

    int *cell_i_j = pitched_pointer(matrix, i, j, pitch); 

    //foreach k: t*B <= t < t+B
    for (int k = BLOCK_START(t,B); k < BLOCK_END(t,B); k++) {

        if (
            /* row index is contained in s.d. block and column index is outside */
            ( BLOCK_START(t,B)<=i<BLOCK_END(t,B) && (j<BLOCK_START(t,B) || j>=BLOCK_END(t,B)) )   ||  

            /* column index is contained in s.d. block and row index is outside */
            ( BLOCK_START(t,B)<=j<BLOCK_END(t,B) && (i<BLOCK_START(t,B) || i>=BLOCK_END(t,B)) ) 
            ) {

            int* cell_k_j = pitched_pointer(matrix, k, j, pitch); //(int *)((char*) matrix + k * pitch) + j;
            int* cell_i_k = pitched_pointer(matrix, i, k, pitch); //(int *)((char*) matrix + i * pitch) + k;
    
            int using_k_path = sum_if_not_infinite(*cell_i_k, *cell_k_j, INF); 
    
            if (using_k_path < *cell_i_j) {
                *cell_i_j = using_k_path;
            }
        }

        //printf("i:%d, j:%d, k:%d\n", i, j, k);

        __syncthreads();

    }
}


__global__ void execute_round_device_v_1_3_phase_3(int *matrix, int n, int t, int B, size_t pitch) {

    // Launched blocks and correspondent position in the matrix

    //  UL  UL  UL  UR  UR
    //  UL  UL  UL  UR  UR
    //  UL  UL  UL  UR  UR
    //  DL  DL  DL  DR  DR
    //  DL  DL  DL  DR  DR

    //  UL  UL  UL  -   UR  UR
    //  UL  UL  UL  -   UR  UR
    //  UL  UL  UL  -   UR  UR  
    //  -   -   -   -   -   - 
    //  DL  DL  DL  -   DR  DR
    //  DL  DL  DL  -   DR  DR

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
           
    // if a thread is under t, add B as row offset to get right position in matrix
    if (blockIdx.x >= t)    i += B; 

    // if a thread is ar right of t, add B as col offset to get right position in matrix
    if (blockIdx.y >= t)    j += B;

    int *cell_i_j = pitched_pointer(matrix, i, j, pitch); 

    //foreach k: t*B <= t < t+B
    for (int k = BLOCK_START(t,B); k < BLOCK_END(t,B); k++) {

        int* cell_k_j = pitched_pointer(matrix, k, j, pitch); //(int *)((char*) matrix + k * pitch) + j;
        int* cell_i_k = pitched_pointer(matrix, i, k, pitch); //(int *)((char*) matrix + i * pitch) + k;

        int using_k_path = sum_if_not_infinite(*cell_i_k, *cell_k_j, INF); 

        if (using_k_path < *cell_i_j) {
            *cell_i_j = using_k_path;
        }

        __syncthreads();
    }
}

