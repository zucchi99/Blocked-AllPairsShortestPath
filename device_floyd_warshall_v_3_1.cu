#include <stdlib.h>
#include <stdio.h>
#include <cassert>
#include <vector>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "include/adj_matrix_utils.cuh"
#include "include/adj_matrix_utils.hpp"
#include "include/cuda_errors_utils.cuh"
#include "include/host_floyd_warshall.hpp"
#include "include/macros.hpp"
#include "include/performance_test.cuh"
#include "include/statistical_test.hpp"
#include "include/lcm.hpp"

#define ARR_MATRIX_INDEX(i,j,n) (i*n+j)
#define ARR_MATRIX_INDEX_TRASP(i,j,n) (i+n*j)

#define SHARED_BANK_N_INT 32
#define ARR_MATRIX_INDEX_BANK_CONFLICT(i, j, n, handle_bank_conflict) (i*n + j + (handle_bank_conflict ? i : 0))
#define ARR_MATRIX_SIZE_BANK_CONFICT(B,handle_bank_conflict) (B*B + (handle_bank_conflict ? (B-1) : 0))

//main device code
void floyd_warshall_blocked_device_v_3_1(int *matrix, int n, int B);

//rounds code
__global__ void execute_round_device_v_3_1_phase_1(int *matrix, int n, int t, bool handle_bank_conflict);

__global__ void execute_round_device_v_3_1_phase_2_row_portion(int *matrix, int n, int t, int start_col, int end_col);
__global__ void execute_round_device_v_3_1_phase_2_col_portion(int *matrix, int n, int t, int start_row, int end_row);
__global__ void execute_round_device_v_3_1_phase_3_portion(int *matrix, int n, int t, int start_row, int start_col, int end_row, int end_col);


int main() {

    MultiSizeTestParameters my_params;
    my_params.f = &floyd_warshall_blocked_device_v_3_1;
    my_params.g = &host_array_floyd_warshall_blocked;
    my_params.start_input_size = 4;
    my_params.end_input_size = 100;
    my_params.costant_multiplier = 1.4;
    my_params.min_blocking_factor = 2;

    print_multi_size_test_parameters(my_params);
    multi_size_statistical_test(my_params);

    return 0;
}

cudaKernelNodeParams cuda_graph_node_params_copy(cudaKernelNodeParams params) {
    
    cudaKernelNodeParams newParams = { 0 };

    newParams.func = params.func;
    newParams.blockDim = params.blockDim;
    newParams.gridDim = params.gridDim;
    newParams.kernelParams = params.kernelParams;
    newParams.sharedMemBytes = params.sharedMemBytes;
    newParams.extra = params.extra;

    return newParams;
}


void floyd_warshall_blocked_device_v_3_1(int *matrix, int n, int B) {

    assert(n%B == 0);                       // B must divide n
    assert(B*B<=MAX_BLOCK_SIZE);            // B*B cannot exceed max block size

    cudaStream_t streams[4];
    for (int i=0; i<4; i++) {
        cudaStreamCreate(&streams[i]);
    }

    int *dev_rand_matrix;
    HANDLE_ERROR(cudaMalloc( (void**) &dev_rand_matrix, n*n*sizeof(int)));
    HANDLE_ERROR(cudaMemcpy(dev_rand_matrix, matrix, n*n*sizeof(int), cudaMemcpyHostToDevice));

    int num_rounds = n/B;

    bool bank_conflict_phase_1 = lcm(SHARED_BANK_N_INT, B) <= (B-1)*B;

         
    for(int t = 0; t < num_rounds; t++) { 

        //arr_execute_round(int *matrix, int n, int t, int row, int col, int B)

        cudaGraph_t roundGraph;
        cudaGraphCreate(&roundGraph, 0);

        std::vector<cudaGraphNode_t> nodeDependencies = {}; // Dependency vector 

        // ----------------------------------------------------------------------
        // phase 1: self-dependent block

        dim3 num_blocks(max(num_rounds-1, 1), max(num_rounds-1, 1));
        dim3 threads_per_block(B, B);

        // execute_round_device_v_3_1_phase_1<<<
        //     num_blocks, 
        //     threads_per_block, 
        //     ARR_MATRIX_SIZE_BANK_CONFICT(B, bank_conflict_phase_1)*sizeof(int), 
        //     streams[0]>>>(dev_rand_matrix, n, t, bank_conflict_phase_1);

        // HANDLE_ERROR(cudaDeviceSynchronize());

        void* phase1_args[4] = { (void*) &dev_rand_matrix, (void*) &n, (void*) &t, (void*) &bank_conflict_phase_1 };

        cudaKernelNodeParams phase1_params;

        phase1_params.func = (void*) execute_round_device_v_3_1_phase_1;
        phase1_params.gridDim = num_blocks;
        phase1_params.blockDim = threads_per_block;
        phase1_params.sharedMemBytes = max(
            ARR_MATRIX_SIZE_BANK_CONFICT(B, bank_conflict_phase_1)*sizeof(int), 
            2*B*B*sizeof(int)
        );
        phase1_params.kernelParams = (void**) phase1_args;
        phase1_params.extra = NULL;

        cudaGraphNode_t phase1_node;

        HANDLE_ERROR(cudaGraphAddKernelNode(
            &phase1_node, roundGraph, 
            nodeDependencies.data(), nodeDependencies.size(), 
            &phase1_params));
        

        // ----------------------------------------------------------------------
        // phase 2: row and cols
        // all blocks that share a row or a column with the self dependent, so
        //  -   all blocks just above or under t
        //  -   all block at left and at right of t


        // execute_round_device_v_3_1_phase_2_col_portion<<<
        //     num_blocks, threads_per_block, 
        //     2*B*B*sizeof(int), 
        //     streams[0]>>>(dev_rand_matrix, n, t, 0, t);

        // execute_round_device_v_3_1_phase_2_row_portion<<<
        //     num_blocks, threads_per_block, 
        //     2*B*B*sizeof(int), 
        //     streams[1]>>>(dev_rand_matrix, n, t, 0, t);

        // execute_round_device_v_3_1_phase_2_col_portion<<<
        //     num_blocks, threads_per_block, 
        //     2*B*B*sizeof(int), 
        //     streams[2]>>>(dev_rand_matrix, n, t, t+1, num_rounds);

        // execute_round_device_v_3_1_phase_2_row_portion<<<
        //     num_blocks, threads_per_block, 
        //     2*B*B*sizeof(int), 
        //     streams[3]>>>(dev_rand_matrix, n, t, t+1, num_rounds);

        // HANDLE_ERROR(cudaDeviceSynchronize()); 

        // nodeDependencies.clear();
        nodeDependencies.push_back(phase1_node);

        // up 
        // execute_round_device_v_3_1_phase_2_col_portion<<<
        //     t, threads_per_block, 
        //     2*B*B*sizeof(int), 
        //     streams[0]>>>(dev_rand_matrix, n, t, 0);

        int start_up_left = 0;
        int end_up_left = t;
        void* phase2_up_left_args[5] = { (void*) &dev_rand_matrix, 
            &n, &t, &start_up_left, &end_up_left };

        cudaKernelNodeParams phase2_up_params = cuda_graph_node_params_copy(phase1_params);

        phase2_up_params.func = (void*) execute_round_device_v_3_1_phase_2_col_portion;
        // phase2_up_params.sharedMemBytes = 2*B*B*sizeof(int);
        phase2_up_params.kernelParams = (void**) phase2_up_left_args;

        cudaGraphNode_t phase2_up_node;

        HANDLE_ERROR(cudaGraphAddKernelNode(
            &phase2_up_node, roundGraph, 
            nodeDependencies.data(), nodeDependencies.size(), 
            &phase2_up_params
        ));

        // left
        // execute_round_device_v_3_1_phase_2_row_portion<<<
        //     t, threads_per_block, 
        //     2*B*B*sizeof(int), 
        //     streams[1]>>>(dev_rand_matrix, n, t, 0);

        cudaKernelNodeParams phase2_left_params = cuda_graph_node_params_copy(phase2_up_params);
        phase2_left_params.func = (void*) execute_round_device_v_3_1_phase_2_row_portion;

        cudaGraphNode_t phase2_left_node;

        HANDLE_ERROR(cudaGraphAddKernelNode(
            &phase2_left_node, roundGraph, 
            nodeDependencies.data(), nodeDependencies.size(), 
            &phase2_left_params));

        // down
        // execute_round_device_v_3_1_phase_2_col_portion<<<
        //     num_rounds-1-t, threads_per_block, 
        //     2*B*B*sizeof(int), 
        //     streams[2]>>>(dev_rand_matrix, n, t, t+1);

        int start_down_right = t+1;
        int end_down_right = num_rounds;

        cudaKernelNodeParams phase2_down_params = cuda_graph_node_params_copy(phase2_up_params);
        void* phase2_down_right_args[5] = { (void*) &dev_rand_matrix, 
            &n, &t, &start_down_right, &end_down_right};
        phase2_down_params.kernelParams = (void**) phase2_down_right_args;

        cudaGraphNode_t phase2_down_node;

        HANDLE_ERROR(cudaGraphAddKernelNode(
            &phase2_down_node, roundGraph, 
            nodeDependencies.data(), nodeDependencies.size(), 
            &phase2_down_params));

        // right
        // execute_round_device_v_3_1_phase_2_row_portion<<<
        //     num_rounds-1-t, threads_per_block, 
        //     2*B*B*sizeof(int), 
        //     streams[3]>>>(dev_rand_matrix, n, t, t+1);

        cudaKernelNodeParams phase2_right_params = cuda_graph_node_params_copy(phase2_down_params);
        phase2_right_params.func = (void*) execute_round_device_v_3_1_phase_2_row_portion;

        cudaGraphNode_t phase2_right_node;

        HANDLE_ERROR(cudaGraphAddKernelNode(
            &phase2_right_node, roundGraph, 
            nodeDependencies.data(), nodeDependencies.size(), 
            &phase2_right_params));

        // HANDLE_ERROR(cudaDeviceSynchronize());

        cudaGraphExec_t instance;
        
        HANDLE_ERROR(cudaGraphInstantiate(&instance, roundGraph, NULL, NULL, 0));

        HANDLE_ERROR(cudaGraphLaunch(instance, streams[1]));
        HANDLE_ERROR(cudaStreamSynchronize(streams[1]));

        // Clean up
        HANDLE_ERROR(cudaGraphExecDestroy(instance));
        HANDLE_ERROR(cudaGraphDestroy(roundGraph));

        HANDLE_ERROR(cudaDeviceSynchronize()); 

        // phase 3: all the remaining blocks, so all the blocks that don't share a row or a col with t

        // dim3 num_blocks_phase_3(num_rounds-1, num_rounds-1); 
        // execute_round_device_v_3_1_phase_3<<<num_blocks_phase_3, threads_per_block, 2*B*B*sizeof(int)>>>(dev_rand_matrix, n, t);

        dim3 num_blocks_phase_3_ul(t, t);
        execute_round_device_v_3_1_phase_3_portion<<<
            num_blocks, threads_per_block, 
            2*B*B*sizeof(int), 
            streams[0]>>>(dev_rand_matrix, n, t, 0, 0, t, t);

        dim3 num_blocks_phase_3_dr(num_rounds-t-1, num_rounds-t-1); 
        execute_round_device_v_3_1_phase_3_portion<<<
            num_blocks, threads_per_block, 
            2*B*B*sizeof(int), 
            streams[1]>>>(dev_rand_matrix, n, t, t+1, t+1, num_rounds, num_rounds);

        dim3 num_blocks_phase_3_ur(t, num_rounds-t-1); 
        execute_round_device_v_3_1_phase_3_portion<<<
            num_blocks, threads_per_block, 
            2*B*B*sizeof(int), 
            streams[2]>>>(dev_rand_matrix, n, t, 0, t+1, t, num_rounds);

        dim3 num_blocks_phase_3_dl(num_rounds-t-1, t); 
        execute_round_device_v_3_1_phase_3_portion<<<
            num_blocks, threads_per_block, 
            2*B*B*sizeof(int), 
            streams[3]>>>(dev_rand_matrix, n, t, t+1, 0, num_rounds, t);

        HANDLE_ERROR(cudaDeviceSynchronize()); 
    }

    // HANDLE_ERROR(cudaDeviceSynchronize());  

    HANDLE_ERROR(cudaMemcpy(matrix, dev_rand_matrix, n*n*sizeof(int), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaFree(dev_rand_matrix));

    for (int i=0; i<4; i++) {
        HANDLE_ERROR(cudaStreamDestroy(streams[i]));
    }
}


__global__ void execute_round_device_v_3_1_phase_1(int *matrix, int n, int t, bool handle_bank_conflict) {

    // Launched block and correspondent position in the matrix

    //  t   -   -   -   -
    //  -   -   -   -   -
    //  -   -   -   -   -
    //  -   -   -   -   -
    //  -   -   -   -   -
    

    //  .   .   .   .   .   . 
    //  .   .   .   .   .   . 
    //  .   .   .   .   .   . 
    //  .   .   .   t   .   .
    //  .   .   .   .   .   . 
    //  .   .   .   .   .   . 

    if (blockIdx.x > 0 || blockIdx.y > 0)   return;

    // if (threadIdx.x == 0 && threadIdx.y == 0) printf("(%d,%d) ", blockIdx.x, blockIdx.y);

    extern __shared__ int block_t_t_shared[];

    int i = threadIdx.x + t * blockDim.x;  // row abs index
    int j = threadIdx.y + t * blockDim.x;  // col abs index

    block_t_t_shared[ARR_MATRIX_INDEX_BANK_CONFLICT(threadIdx.x, threadIdx.y, blockDim.x, handle_bank_conflict)] = matrix[ARR_MATRIX_INDEX(i, j, n)];

    __syncthreads();

    // now k is iterating the relative indexind of (t,t) block 
    // in shared memory (instead of the abs position in matrix)
    for (int k = 0; k < blockDim.x; k++) {

        int using_k_path = sum_if_not_infinite(
            block_t_t_shared[ARR_MATRIX_INDEX_BANK_CONFLICT(threadIdx.x, k, blockDim.x, handle_bank_conflict)], 
            block_t_t_shared[ARR_MATRIX_INDEX_BANK_CONFLICT(k, threadIdx.y, blockDim.x, handle_bank_conflict)], 
            INF
        ); 

        if (using_k_path < block_t_t_shared[ARR_MATRIX_INDEX_BANK_CONFLICT(threadIdx.x, threadIdx.y, blockDim.x, handle_bank_conflict)]) {
            block_t_t_shared[ARR_MATRIX_INDEX_BANK_CONFLICT(threadIdx.x, threadIdx.y, blockDim.x, handle_bank_conflict)] = using_k_path;
        }
        
        __syncthreads();
    }

    matrix[ARR_MATRIX_INDEX(i, j, n)] = block_t_t_shared[ARR_MATRIX_INDEX_BANK_CONFLICT(threadIdx.x, threadIdx.y, blockDim.x, handle_bank_conflict)];
}


__global__ void execute_round_device_v_3_1_phase_2_row_portion(int *matrix, int n, int t, int start_col, int end_col) {
    
    if (blockIdx.x >= end_col-start_col)    return;
    
    extern __shared__ int shared_mem[];
    
    int* block_i_j_shared = &shared_mem[0];
    int* block_t_t_shared = &shared_mem[(blockDim.x * blockDim.x)];

    // it's a row ...

    // abs row index 
    int i = BLOCK_START(t, blockDim.x) + threadIdx.x;    
    // abs col index   
    int j = BLOCK_START(blockIdx.x, blockDim.x) + threadIdx.y + start_col * blockDim.x; 

    // the block where I am working
    block_i_j_shared[ARR_MATRIX_INDEX(threadIdx.x, threadIdx.y, blockDim.x)] = matrix[ARR_MATRIX_INDEX(i, j, n)];

    // the self-dependent block already calculated in this round (transposed to avoid bank conflict)
    block_t_t_shared[ARR_MATRIX_INDEX_TRASP(threadIdx.x, threadIdx.y, blockDim.x)] = matrix[
        ARR_MATRIX_INDEX(
            (BLOCK_START(t, blockDim.x) + threadIdx.x), 
            (BLOCK_START(t, blockDim.x) + threadIdx.y), 
            n
        )
    ];


    __syncthreads();

    // now k is iterating the relative indexind of (t,t) block 
    // in shared memory (instead of the abs position in matrix)
    for (int k = 0; k < blockDim.x; k++) {

        // Because we are doing rows:
        // -    matrix[i,abs_k] is in block_t_t_shared[threadIdx.x,k]
        // -    matrix[abs_k,j] is in block_i_j_shared[k,threadIdx.y]
        int using_k_path = sum_if_not_infinite(
            block_t_t_shared[ARR_MATRIX_INDEX_TRASP(threadIdx.x, k, blockDim.x)], 
            block_i_j_shared[ARR_MATRIX_INDEX(k, threadIdx.y, blockDim.x)], 
            INF
        ); 

        if (using_k_path < block_i_j_shared[ARR_MATRIX_INDEX(threadIdx.x, threadIdx.y, blockDim.x)]) {
            block_i_j_shared[ARR_MATRIX_INDEX(threadIdx.x, threadIdx.y, blockDim.x)] = using_k_path;
        }

        //printf("i:%d, j:%d, k:%d\n", i, j, k);

        __syncthreads();
    }

    // copy result in global memory
    matrix[ARR_MATRIX_INDEX(i, j, n)] = block_i_j_shared[ARR_MATRIX_INDEX(threadIdx.x, threadIdx.y, blockDim.x)];
}


__global__ void execute_round_device_v_3_1_phase_2_col_portion(int *matrix, int n, int t, int start_row, int end_row) {
    
    if (blockIdx.x >= end_row-start_row)    return;
    
    extern __shared__ int shared_mem[];

    int* block_i_j_shared = &shared_mem[0];
    int* block_t_t_shared = &shared_mem[blockDim.x*blockDim.x];

    // it's a column ...

    // abs row index 
    int i = BLOCK_START(blockIdx.x, blockDim.x) + threadIdx.x + start_row * blockDim.x;
    // abs col index 
    int j = BLOCK_START(t, blockDim.x) + threadIdx.y;

    // the block where I am working (transposed to avoid bank conflict)
    block_i_j_shared[ARR_MATRIX_INDEX_TRASP(threadIdx.x, threadIdx.y, blockDim.x)] = matrix[ARR_MATRIX_INDEX(i, j, n)];

    // the self-dependent block already calculated in this round 
    block_t_t_shared[ARR_MATRIX_INDEX(threadIdx.x, threadIdx.y, blockDim.x)] = matrix[
        ARR_MATRIX_INDEX(
            (BLOCK_START(t, blockDim.x) + threadIdx.x), 
            (BLOCK_START(t, blockDim.x) + threadIdx.y), 
            n
        )
    ];
    
    __syncthreads();

    // now k is iterating the relative indexind of (t,t) block 
    // in shared memory (instead of the abs position in matrix)
    for (int k = 0; k < blockDim.x; k++) {
        
        // Because we are doing columns:
        // -    matrix[i,k] is in block_i_j_shared[threadIdx.x,k]
        // -    matrix[k,j] is in block_t_t_shared[k,threadIdx.y]
        int using_k_path = sum_if_not_infinite(
            block_i_j_shared[ARR_MATRIX_INDEX_TRASP(threadIdx.x, k, blockDim.x)], 
            block_t_t_shared[ARR_MATRIX_INDEX(k, threadIdx.y, blockDim.x)], 
            INF
        ); 

        if (using_k_path < block_i_j_shared[ARR_MATRIX_INDEX_TRASP(threadIdx.x, threadIdx.y, blockDim.x)]) {
            block_i_j_shared[ARR_MATRIX_INDEX_TRASP(threadIdx.x, threadIdx.y, blockDim.x)] = using_k_path;
        }

        //printf("i:%d, j:%d, k:%d\n", i, j, k);

        __syncthreads();
    }

    // copy result in global memory
    matrix[ARR_MATRIX_INDEX(i, j, n)] = block_i_j_shared[ARR_MATRIX_INDEX_TRASP(threadIdx.x, threadIdx.y, blockDim.x)];
}



__global__ void execute_round_device_v_3_1_phase_3_portion(int *matrix, int n, int t, int start_row, int start_col, int end_row, int end_col) {

    if (blockIdx.x >= end_row-start_row || blockIdx.y >= end_col-start_col)    return;
    
    extern __shared__ int shared_mem[];

    int* block_i_t_shared = &shared_mem[0];
    int* block_t_j_shared = &shared_mem[blockDim.x*blockDim.x];

    // abs row index
    int i = threadIdx.x + blockIdx.x * blockDim.x + start_row * blockDim.x;
    // abs col index
    int j = threadIdx.y + blockIdx.y * blockDim.y + start_col * blockDim.y;

    // printf("%d,%d\n",i,j);
    
    // since the cell i,j is read and written only by this thread
    // there is no need to copy its value to shared memory we can just us a local variable
    int cell_i_j = matrix[ARR_MATRIX_INDEX(i, j, n)];
        
    // In phase 3 I copy in two portions of my shared memory
    // the block corresponding to (t, this column) and (this row, t). 

    // (this row, t) is transposed to prevent bank conflict

    block_i_t_shared[ARR_MATRIX_INDEX_TRASP(threadIdx.x, threadIdx.y, blockDim.x)] = matrix[
        ARR_MATRIX_INDEX(i, (BLOCK_START(t, blockDim.x) + threadIdx.y), n)
    ];
    block_t_j_shared[ARR_MATRIX_INDEX(threadIdx.x, threadIdx.y, blockDim.x)] = matrix[
        ARR_MATRIX_INDEX((BLOCK_START(t, blockDim.x) + threadIdx.x), j, n)
    ];
    
    __syncthreads();

    // now k is iterating the relative indexind of (t,t) block 
    // in shared memory (instead of the abs position in matrix)
    for (int k = 0; k < blockDim.x; k++) {

        int using_k_path = sum_if_not_infinite(
            block_i_t_shared[ARR_MATRIX_INDEX_TRASP(threadIdx.x, k, blockDim.x)],
            block_t_j_shared[ARR_MATRIX_INDEX(k, threadIdx.y, blockDim.x)],
            INF
        ); 

        if (using_k_path < cell_i_j) {
            cell_i_j = using_k_path;
        }

        __syncthreads();
    }

    // copy result in global memory
    matrix[ARR_MATRIX_INDEX(i, j, n)] = cell_i_j;
}
