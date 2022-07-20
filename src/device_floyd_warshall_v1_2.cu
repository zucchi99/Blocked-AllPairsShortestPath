#include "../include/device_floyd_warshall_v1_2.cuh"

#include <cassert>

#include "../include/cuda_errors_utils.cuh"
#include "../include/num_macro.hpp"


void floyd_warshall_blocked_device_v1_2(int *matrix, int n, int B) {

    assert(n%B == 0);                       // B must divide n
    assert(B*B<=MAX_BLOCK_SIZE);            // B*B cannot exceed max block size

    int *dev_rand_matrix;
    HANDLE_ERROR(cudaMalloc( (void**) &dev_rand_matrix, n * n* sizeof(int)));
    HANDLE_ERROR(cudaMemcpy(dev_rand_matrix, matrix, n*n*sizeof(int), cudaMemcpyHostToDevice));

    int num_rounds = n/B;
     
    for(int t = 0; t < num_rounds; t++) { 

        //arr_execute_round(int *matrix, int n, int t, int row, int col, int B)

        //phase 1: self-dependent block
        dim3 num_blocks_phase_1(1, 1);
        dim3 threads_per_block_phase_1(B, B);

        execute_round_device_v1_2_phase_1<<<num_blocks_phase_1, threads_per_block_phase_1>>>(dev_rand_matrix, n, t, B);
        HANDLE_ERROR(cudaDeviceSynchronize());

        // phase 2: all blocks that share a row or a column with the self dependent, so
        //  -   all blocks just above or under t
        //  -   all block at left and at right of t

        // Phase 2/3 thread matrix is made by n*n threads, divided in num_rounds*num_rounds blocks
        dim3 num_blocks_phase_2_3(num_rounds, num_rounds);  

        execute_round_device_v1_2_phase_2<<<num_blocks_phase_2_3, threads_per_block_phase_1>>>(dev_rand_matrix, n, t, B);
        HANDLE_ERROR(cudaDeviceSynchronize());

        // phase 3: all the remaining blocks, so all the blocks that don't share a row or a col with t

        execute_round_device_v1_2_phase_3<<<num_blocks_phase_2_3, threads_per_block_phase_1>>>(dev_rand_matrix, n, t, B);
        HANDLE_ERROR(cudaDeviceSynchronize()); 
    }

    // HANDLE_ERROR(cudaDeviceSynchronize());  

    HANDLE_ERROR(cudaMemcpy(matrix, dev_rand_matrix, n*n*sizeof(int), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaFree(dev_rand_matrix));
}

__global__ void execute_round_device_v1_2_phase_1(int *matrix, int n, int t, int B) {

    int tid_x = threadIdx.x + blockIdx.x * blockDim.x;
    int tid_y = threadIdx.y + blockIdx.y * blockDim.y;

    int i = tid_x + t * B;  // row
    int j = tid_y + t * B;  // col

    //foreach k: t*B <= t < t+B
    for (int k = BLOCK_START(t,B); k < BLOCK_END(t,B); k++) {

        int b = sum_if_not_infinite(matrix[i*n + k], matrix[k*n + j], INF); 

        if (b < matrix[i*n + j]) {
            matrix[i*n + j] = b;
        }
        
        __syncthreads();
    }
}

__global__ void execute_round_device_v1_2_phase_2(int *matrix, int n, int t, int B) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    //foreach k: t*B <= t < t+B
    for (int k = BLOCK_START(t,B); k < BLOCK_END(t,B); k++) {

        if (
            /* row index is contained in s.d. block and column index is outside */
            ( BLOCK_START(t,B)<=i<BLOCK_END(t,B) && (j<BLOCK_START(t,B) || j>=BLOCK_END(t,B)) )   ||  

            /* column index is contained in s.d. block and row index is outside */
            ( BLOCK_START(t,B)<=j<BLOCK_END(t,B) && (i<BLOCK_START(t,B) || i>=BLOCK_END(t,B)) ) 
            ) {

            int b = sum_if_not_infinite(matrix[i*n + k], matrix[k*n + j], INF); 

            if (b < matrix[i*n + j]) {
                matrix[i*n + j] = b;
            }
        }

        __syncthreads();
    }
}

__global__ void execute_round_device_v1_2_phase_3(int *matrix, int n, int t, int B) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    //foreach k: t*B <= t < t+B
    for (int k = BLOCK_START(t,B); k < BLOCK_END(t,B); k++) {

        if (
            /* above and right or left */
            ( i>=BLOCK_END(t,B) && (j<BLOCK_START(t,B) || j>=BLOCK_END(t,B)) )   ||  

            /* under and right or left */
            ( i<BLOCK_START(t,B) && (j<BLOCK_START(t,B) || j>=BLOCK_END(t,B)) ) 
            ) {

            int b = sum_if_not_infinite(matrix[i*n + k], matrix[k*n + j], INF); 

            if (b < matrix[i*n + j]) {
                matrix[i*n + j] = b;
            }
        }

        __syncthreads();
    }
}

