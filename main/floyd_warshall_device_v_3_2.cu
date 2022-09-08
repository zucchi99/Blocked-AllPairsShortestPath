#include "../include/include_needed_libraries.cuh"
#include "../include/lcm.hpp"
#include <vector>

#define ARR_MATRIX_INDEX(i,j,n) (i*n+j)
#define ARR_MATRIX_INDEX_TRASP(i,j,n) (i+n*j)

#define SHARED_BANK_N_INT 32
#define ARR_MATRIX_INDEX_BANK_CONFLICT(i, j, n, handle_bank_conflict) (i*n + j + (handle_bank_conflict ? i : 0))
#define ARR_MATRIX_SIZE_BANK_CONFICT(B,handle_bank_conflict) (B*B + (handle_bank_conflict ? (B-1) : 0))

//main device code
void floyd_warshall_blocked_device_v_3_2(int *matrix, int n, int B);

//rounds code
__global__ void execute_round_device_v_3_2_phase_1(int *matrix, int n, int t, bool handle_bank_conflict);

__global__ void execute_round_device_v_3_2_phase_2_row_portion(int *matrix, int n, int t, int start_col, int end_col);
__global__ void execute_round_device_v_3_2_phase_2_col_portion(int *matrix, int n, int t, int start_row, int end_row);
__global__ void execute_round_device_v_3_2_phase_3_portion(int *matrix, int n, int t, int start_row, int start_col, int end_row, int end_col);


int main(int argc, char *argv[]) {

    return handle_arguments_and_execute(argc, argv, (void(*) (int*, int, int)) &floyd_warshall_blocked_device_v_3_2);
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

cudaMemcpy3DParms cuda_graph_get_memcpy_params(
    int* src, int* dst, int n, cudaMemcpyKind kind) {
    
    cudaMemcpy3DParms copy_params = {0};

    copy_params.srcArray = NULL;
    copy_params.srcPos = make_cudaPos(0, 0, 0);
    copy_params.srcPtr = make_cudaPitchedPtr((void*) src, n*sizeof(int), n, n);
    copy_params.dstArray = NULL;
    copy_params.dstPos = make_cudaPos(0, 0, 0);
    copy_params.dstPtr = make_cudaPitchedPtr((void*) dst, n*sizeof(int), n, n);
    copy_params.extent = make_cudaExtent(n*sizeof(int), n, 1);
    copy_params.kind = kind;

    return copy_params;
}


void floyd_warshall_blocked_device_v_3_2(int *matrix, int n, int B) {

    assert(n%B == 0);                       // B must divide n
    assert(B*B<=MAX_BLOCK_SIZE);            // B*B cannot exceed max block size

    // init the graph i will use to do all rounds
    cudaGraph_t graph;
    cudaGraphCreate(&graph, 0);
    std::vector<cudaGraphNode_t> nodeDependencies = {}; // Dependency vector 

    int *dev_matrix;
    HANDLE_ERROR(cudaMalloc( (void**) &dev_matrix, n*n*sizeof(int)));

    // HANDLE_ERROR(cudaMemcpy(dev_matrix, matrix, n*n*sizeof(int), cudaMemcpyHostToDevice));

    cudaMemcpy3DParms copy_host_to_dev_params = cuda_graph_get_memcpy_params(
        matrix, dev_matrix, n, cudaMemcpyHostToDevice);

    cudaGraphNode_t copy_host_to_dev_node;

    HANDLE_ERROR(cudaGraphAddMemcpyNode(
        &copy_host_to_dev_node, graph, 
        nodeDependencies.data(), nodeDependencies.size(), 
        &copy_host_to_dev_params
        ));


    // number of rounds that will be executed
    int num_rounds = n/B;

    // check if there will be bank conflict in phase 1
    bool bank_conflict_phase_1 = lcm(SHARED_BANK_N_INT, B) <= (B-1)*B;

    // number of threads launched at each kernel 
    // (it has to be the same number because of the cuda graphs, 
    // the exceeding threads will end as firtst instruction)
    dim3 num_blocks(max(num_rounds-1, 1), max(num_rounds-1, 1));
    dim3 threads_per_block(B, B);

    // a variable needed for calls which requires pointer args
    int zero = 0;

    // previous round nodes (so i can add them as dependency for the next one)
    cudaGraphNode_t prev_phase3_up_left_node,   prev_phase3_up_right_node, 
                    prev_phase3_down_left_node, prev_phase3_down_right_node;

    // last round phase 1 and 2 nodes (used just to save last round nodes and use them as 
    // memcpy device to host dependencies)
    cudaGraphNode_t prev_phase1_node, prev_phase2_up_node, prev_phase2_left_node;
         
    for(int t = 0; t < num_rounds; t++) { 

        // a variable needed for calls which requires pointer args
        int t_plus_1 = t+1;

        // ----------------------------------------------------------------------
        // phase 1: self-dependent block

        // execute_round_device_v_3_2_phase_1<<<
        //     num_blocks, 
        //     threads_per_block, 
        //     ARR_MATRIX_SIZE_BANK_CONFICT(B, bank_conflict_phase_1)*sizeof(int), 
        //     graph_stream>>>(dev_matrix, n, t, bank_conflict_phase_1);

        // HANDLE_ERROR(cudaDeviceSynchronize());

        void* phase1_args[4] = { (void*) &dev_matrix, (void*) &n, (void*) &t, (void*) &bank_conflict_phase_1 };

        cudaKernelNodeParams phase1_params;

        phase1_params.func = (void*) execute_round_device_v_3_2_phase_1;
        phase1_params.gridDim = num_blocks;
        phase1_params.blockDim = threads_per_block;
        phase1_params.sharedMemBytes = max(
            ARR_MATRIX_SIZE_BANK_CONFICT(B, bank_conflict_phase_1)*sizeof(int), 
            2*B*B*sizeof(int)
        );
        phase1_params.sharedMemBytes = ARR_MATRIX_SIZE_BANK_CONFICT(B, bank_conflict_phase_1)*sizeof(int);
        phase1_params.kernelParams = (void**) phase1_args;
        phase1_params.extra = NULL;

        nodeDependencies.clear(); 
        if (t > 0) {
            // round after first should depend from previous phase 3 down-right
            nodeDependencies.push_back(prev_phase3_down_right_node);
        } else {

            // first round phase 1 should depend on memcpy
            nodeDependencies.push_back(copy_host_to_dev_node);
        }

        cudaGraphNode_t phase1_node;

        HANDLE_ERROR(cudaGraphAddKernelNode(
            &phase1_node, graph, 
            nodeDependencies.data(), nodeDependencies.size(), 
            &phase1_params));

        // HANDLE_ERROR(cudaDeviceSynchronize());

        // ----------------------------------------------------------------------
        // phase 2: row and cols
        // all blocks that share a row or a column with the self dependent, so
        //  -   all blocks just above or under t
        //  -   all block at left and at right of t

        // up 
        // execute_round_device_v_3_2_phase_2_col_portion<<<
        //     num_blocks, threads_per_block, 
        //     2*B*B*sizeof(int), 
        //     graph_stream>>>(dev_matrix, n, t, 0, t);

        void* phase2_up_left_args[5] = { (void*) &dev_matrix, 
            &n, &t, &zero, &t };

        cudaKernelNodeParams phase2_up_params = cuda_graph_node_params_copy(phase1_params);

        phase2_up_params.func = (void*) execute_round_device_v_3_2_phase_2_col_portion;
        phase2_up_params.sharedMemBytes = 2*B*B*sizeof(int);
        phase2_up_params.kernelParams = (void**) phase2_up_left_args;

        nodeDependencies.clear();
        nodeDependencies.push_back(phase1_node);
        if (t>0) {
            // phase 2 up of rounds after first should depend on previous phase 3 up-right
            nodeDependencies.push_back(prev_phase3_up_right_node);
        }

        cudaGraphNode_t phase2_up_node;

        HANDLE_ERROR(cudaGraphAddKernelNode(
            &phase2_up_node, graph, 
            nodeDependencies.data(), nodeDependencies.size(), 
            &phase2_up_params
        ));

        // left
        // execute_round_device_v_3_2_phase_2_row_portion<<<
        //     num_blocks, threads_per_block, 
        //     2*B*B*sizeof(int), 
        //     graph_stream>>>(dev_matrix, n, t, 0, t);

        cudaKernelNodeParams phase2_left_params = cuda_graph_node_params_copy(phase2_up_params);
        phase2_left_params.func = (void*) execute_round_device_v_3_2_phase_2_row_portion;

        nodeDependencies.clear();
        nodeDependencies.push_back(phase1_node);
        if (t>0) {
            // phase 2 left of rounds after first should depend on previous phase 3 down-left
            nodeDependencies.push_back(prev_phase3_down_left_node);
        }

        cudaGraphNode_t phase2_left_node;

        HANDLE_ERROR(cudaGraphAddKernelNode(
            &phase2_left_node, graph, 
            nodeDependencies.data(), nodeDependencies.size(), 
            &phase2_left_params));

        // down
        // execute_round_device_v_3_2_phase_2_col_portion<<<
        //     num_blocks, threads_per_block, 
        //     2*B*B*sizeof(int), 
        //     graph_stream>>>(dev_matrix, n, t, t+1, num_blocks);

        cudaKernelNodeParams phase2_down_params = cuda_graph_node_params_copy(phase2_up_params);
        void* phase2_down_right_args[5] = { (void*) &dev_matrix, 
            &n, &t, &t_plus_1, &num_rounds};
        phase2_down_params.kernelParams = (void**) phase2_down_right_args;

        nodeDependencies.clear();
        nodeDependencies.push_back(phase1_node);

        // if (t==0) {
        //     // first round phase 2 down should depend on memcpy
        //     nodeDependencies.push_back(copy_host_to_dev_node);
        // }

        cudaGraphNode_t phase2_down_node;

        HANDLE_ERROR(cudaGraphAddKernelNode(
            &phase2_down_node, graph, 
            nodeDependencies.data(), nodeDependencies.size(), 
            &phase2_down_params));

        // right
        // execute_round_device_v_3_2_phase_2_row_portion<<<
        //     num_blocks, threads_per_block, 
        //     2*B*B*sizeof(int), 
        //     graph_stream>>>(dev_matrix, n, t, t+1, num_blocks);

        cudaKernelNodeParams phase2_right_params = cuda_graph_node_params_copy(phase2_down_params);
        phase2_right_params.func = (void*) execute_round_device_v_3_2_phase_2_row_portion;

        nodeDependencies.clear();
        nodeDependencies.push_back(phase1_node);

        // if (t==0) {
        //     // first round phase 2 right should depend on memcpy
        //     nodeDependencies.push_back(copy_host_to_dev_node);
        // }

        cudaGraphNode_t phase2_right_node;

        HANDLE_ERROR(cudaGraphAddKernelNode(
            &phase2_right_node, graph, 
            nodeDependencies.data(), nodeDependencies.size(), 
            &phase2_right_params));

        // HANDLE_ERROR(cudaDeviceSynchronize());
        
        // phase 3: all the remaining blocks, so all the blocks that don't share a row or a col with t
        
        // up-left
        // execute_round_device_v_3_2_phase_3_portion<<<
        //     num_blocks, threads_per_block, 
        //     2*B*B*sizeof(int), 
        //     graph_stream>>>(dev_matrix, n, t, 0, 0, t, t);

        void* phase3_up_left_args[7] = {(void*) &dev_matrix, 
            &n, &t, &zero, &zero, &t, &t};

        cudaKernelNodeParams phase3_up_left_params = cuda_graph_node_params_copy(phase2_up_params);

        phase3_up_left_params.func = (void*) execute_round_device_v_3_2_phase_3_portion;
        phase3_up_left_params.kernelParams = (void**) phase3_up_left_args;

        nodeDependencies.clear();
        nodeDependencies.push_back(phase2_up_node);
        nodeDependencies.push_back(phase2_left_node);

        if (t>0) {
            // phase 3 up-left of rounds after first should depend on previous phase 3 up-left
            nodeDependencies.push_back(prev_phase3_up_left_node);
        }

        cudaGraphNode_t phase3_up_left_node;

        HANDLE_ERROR(cudaGraphAddKernelNode(
            &phase3_up_left_node, graph, 
            nodeDependencies.data(), nodeDependencies.size(), 
            &phase3_up_left_params));

        // up-right
        // execute_round_device_v_3_2_phase_3_portion<<<
        //     num_blocks, threads_per_block, 
        //     2*B*B*sizeof(int), 
        //     graph_stream>>>(dev_matrix, n, t, 0, t+1, t, num_rounds);

        void* phase3_up_right_args[7] = {(void*) &dev_matrix, 
            &n, &t, &zero, &t_plus_1, &t, &num_rounds};

        cudaKernelNodeParams phase3_up_right_params = cuda_graph_node_params_copy(phase3_up_left_params);
        phase3_up_right_params.kernelParams = (void**) phase3_up_right_args;

        nodeDependencies.clear();
        nodeDependencies.push_back(phase2_up_node);
        nodeDependencies.push_back(phase2_right_node);

        cudaGraphNode_t phase3_up_right_node;

        HANDLE_ERROR(cudaGraphAddKernelNode(
            &phase3_up_right_node, graph, 
            nodeDependencies.data(), nodeDependencies.size(), 
            &phase3_up_right_params));

        // down-right
        // execute_round_device_v_3_2_phase_3_portion<<<
        //     num_blocks, threads_per_block, 
        //     2*B*B*sizeof(int), 
        //     graph_stream>>>(dev_matrix, n, t, t+1, t+1, num_rounds, num_rounds);

        void* phase3_down_right_args[7] = {(void*) &dev_matrix, 
            &n, &t, &t_plus_1, &t_plus_1, &num_rounds, &num_rounds};

        cudaKernelNodeParams phase3_down_right_params = cuda_graph_node_params_copy(phase3_up_left_params);
        phase3_down_right_params.kernelParams = (void**) phase3_down_right_args;

        nodeDependencies.clear();
        nodeDependencies.push_back(phase2_down_node);
        nodeDependencies.push_back(phase2_right_node);

        // if (t==0) {
        //     // first round phase 3 down-right should depend on memcpy
        //     nodeDependencies.push_back(copy_host_to_dev_node);
        // }

        cudaGraphNode_t phase3_down_right_node;

        HANDLE_ERROR(cudaGraphAddKernelNode(
            &phase3_down_right_node, graph, 
            nodeDependencies.data(), nodeDependencies.size(), 
            &phase3_down_right_params));

        // down-left
        // execute_round_device_v_3_2_phase_3_portion<<<
        //     num_blocks, threads_per_block, 
        //     2*B*B*sizeof(int), 
        //     graph_stream>>>(dev_matrix, n, t, t+1, 0, num_rounds, t);
        
        void* phase3_down_left_args[7] = {(void*) &dev_matrix, 
            &n, &t, &t_plus_1, &zero, &num_rounds, &t};

        cudaKernelNodeParams phase3_down_left_params = cuda_graph_node_params_copy(phase3_up_left_params);
        phase3_down_left_params.kernelParams = (void**) phase3_down_left_args;

        nodeDependencies.clear();
        nodeDependencies.push_back(phase2_down_node);
        nodeDependencies.push_back(phase2_left_node);

        cudaGraphNode_t phase3_down_left_node;

        HANDLE_ERROR(cudaGraphAddKernelNode(
            &phase3_down_left_node, graph, 
            nodeDependencies.data(), nodeDependencies.size(), 
            &phase3_down_left_params));

        // HANDLE_ERROR(cudaDeviceSynchronize());   

        // save phase 3 nodes
        prev_phase3_up_left_node    = phase3_up_left_node;
        prev_phase3_up_right_node   = phase3_up_right_node;
        prev_phase3_down_right_node = phase3_down_right_node;
        prev_phase3_down_left_node  = phase3_down_left_node;

        if (t==num_rounds-1) {
            // if this is last round, I save previous phase 1 and 2 nodes
            // (needed as dependencies of memcpy operations)
            prev_phase1_node        = phase1_node;
            prev_phase2_left_node   = phase2_left_node;
            prev_phase2_up_node     = phase2_up_node;
        }
    }


    // Add copy of final result from device to host (as graph)
    // HANDLE_ERROR(cudaMemcpy(matrix, dev_matrix, n*n*sizeof(int), cudaMemcpyDeviceToHost));

    cudaMemcpy3DParms copy_dev_to_host_params = cuda_graph_get_memcpy_params(
        dev_matrix, matrix, n, cudaMemcpyDeviceToHost);

    nodeDependencies.clear();
    nodeDependencies.push_back(prev_phase1_node);
    nodeDependencies.push_back(prev_phase2_left_node);
    nodeDependencies.push_back(prev_phase2_up_node);
    nodeDependencies.push_back(prev_phase3_up_left_node);

    cudaGraphNode_t copy_dev_to_host_node;

    HANDLE_ERROR(cudaGraphAddMemcpyNode(
        &copy_dev_to_host_node, graph, 
        nodeDependencies.data(), nodeDependencies.size(), 
        &copy_dev_to_host_params
        ));

    // stream used for executing graph
    cudaStream_t graph_stream;
    cudaStreamCreate(&graph_stream);

    // Instanciate and run graph
    cudaGraphExec_t instance;
    HANDLE_ERROR(cudaGraphInstantiate(&instance, graph, NULL, NULL, 0));
    HANDLE_ERROR(cudaGraphLaunch(instance, graph_stream));
    HANDLE_ERROR(cudaStreamSynchronize(graph_stream));

    // Clean up
    HANDLE_ERROR(cudaGraphExecDestroy(instance));
    HANDLE_ERROR(cudaGraphDestroy(graph));

    // HANDLE_ERROR(cudaDeviceSynchronize());  

    HANDLE_ERROR(cudaFree(dev_matrix));

    HANDLE_ERROR(cudaStreamDestroy(graph_stream));

}


__global__ void execute_round_device_v_3_2_phase_1(int *matrix, int n, int t, bool handle_bank_conflict) {

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


__global__ void execute_round_device_v_3_2_phase_2_row_portion(int *matrix, int n, int t, int start_col, int end_col) {
    
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


__global__ void execute_round_device_v_3_2_phase_2_col_portion(int *matrix, int n, int t, int start_row, int end_row) {
    
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



__global__ void execute_round_device_v_3_2_phase_3_portion(int *matrix, int n, int t, int start_row, int start_col, int end_row, int end_col) {

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

    }

    // copy result in global memory
    matrix[ARR_MATRIX_INDEX(i, j, n)] = cell_i_j;
}
