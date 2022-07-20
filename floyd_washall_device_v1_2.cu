#include <stdlib.h>
#include <cassert>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "include/cuda_errors_utils.cuh"
#include "include/adj_matrix_utils.hpp"
#include "include/host_floyd_warshall.hpp"
#include "include/statistical_test.hpp"
#include "include/num_macro.hpp"
#include "include/device_floyd_warshall_v1_2.cuh"

int main() {

    multi_size_statistical_test(&floyd_warshall_blocked_device_v1_2, 8, 256, 8, 32, 1000, RANDOM_SEED, false, true);

    return 0;
}

__global__ void execute_round_device_v1_3_phase_2(int *matrix, int n, int t, int B) {

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

__global__ void execute_round_device_v1_3_phase_3(int *matrix, int n, int t, int B) {

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

