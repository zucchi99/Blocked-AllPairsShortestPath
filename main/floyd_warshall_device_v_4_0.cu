#include "../include/include_needed_libraries.cuh"
#include "../include/lcm.hpp"
#include <vector>

#define ARR_MATRIX_INDEX(i,j,n) (i*n+j)
#define ARR_MATRIX_INDEX_TRASP(i,j,n) (i+n*j)

#define SHARED_BANK_N_INT 32
#define ARR_MATRIX_INDEX_BANK_CONFLICT(i, j, n, handle_bank_conflict) (i*n + j + (handle_bank_conflict ? i : 0))
#define ARR_MATRIX_SIZE_BANK_CONFICT(B,handle_bank_conflict) (B*B + (handle_bank_conflict ? (B-1) : 0))

//main device code
void floyd_warshall_blocked_device_v_4_0(int *matrix, int n, int B);

//rounds code
__global__ void execute_round_device_v_4_0_phase_1(int *matrix, int n, int t, bool handle_bank_conflict);

__global__ void execute_round_device_v_4_0_phase_2_row_portion(int *matrix, int n, int t, int start_col, int end_col);
__global__ void execute_round_device_v_4_0_phase_2_col_portion(int *matrix, int n, int t, int start_row, int end_row);
__global__ void execute_round_device_v_4_0_phase_3_portion(int *matrix, int n, int t, int start_row, int start_col, int end_row, int end_col);


int main(int argc, char *argv[]) {

    return handle_arguments_and_execute(argc, argv, (void(*) (int*, int, int)) &floyd_warshall_blocked_device_v_4_0);
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


void floyd_warshall_blocked_device_v_4_0(int *matrix, int n, int B) {

    assert(n%B == 0);                       // B must divide n
    assert(B*B<=MAX_BLOCK_SIZE);            // B*B cannot exceed max block size
    
    // matrix data, malloc matrix to device
    int *dev_matrix;
    HANDLE_ERROR(cudaMalloc( (void**) &dev_matrix, n*n*sizeof(int)));
    
    // number of rounds that will be executed
    int num_rounds = n/B;

    // check if there will be bank conflict in phase 1
    bool bank_conflict_phase_1 = lcm(SHARED_BANK_N_INT, B) <= (B-1)*B;

    // number of threads launched at each kernel 
    // (it has to be the same number because of the cuda graphs, 
    // the exceeding threads will end as first instruction)
    dim3 num_blocks(max(num_rounds-1, 1), max(num_rounds-1, 1));
    dim3 threads_per_block(B, B);
    
    // --------------------------------------------------------------------------------------------

    // START GRAPH DEFINITION
    
    // START MEMCPY HOST->DEVICE NODE & DEPENDENCIES DEFINITION

    // init the graph i will use to do all rounds
    cudaGraph_t graph;
    cudaGraphCreate(&graph, 0);
    // vector of nodes dependencies, used as temp vector per each node
    std::vector<cudaGraphNode_t> nodeDependencies = {}; 
    
    // first node: start with memcpy host -> device
    cudaGraphNode_t copy_host_to_dev_node;
    // node parameters
    cudaMemcpy3DParms copy_host_to_dev_params = {0};
    copy_host_to_dev_params.srcArray = NULL;
    copy_host_to_dev_params.srcPos = make_cudaPos(0, 0, 0);
    copy_host_to_dev_params.srcPtr = make_cudaPitchedPtr((void*) matrix, n*sizeof(int), n, n);
    copy_host_to_dev_params.dstArray = NULL;
    copy_host_to_dev_params.dstPos = make_cudaPos(0, 0, 0);
    copy_host_to_dev_params.dstPtr = make_cudaPitchedPtr((void*) dev_matrix, n*sizeof(int), n, n);
    copy_host_to_dev_params.extent = make_cudaExtent(n*sizeof(int), n, 1);
    copy_host_to_dev_params.kind = cudaMemcpyHostToDevice;
    // add node to graph with its dependencies
    HANDLE_ERROR(cudaGraphAddMemcpyNode(
        &copy_host_to_dev_node, graph, 
        nodeDependencies.data(), nodeDependencies.size(), 
        &copy_host_to_dev_params
    ));

    // END MEMCPY HOST->DEVICE NODE & DEPENDENCIES DEFINITION

    // -------------------------------------------------------------------------------

    // previous round nodes (so i can add them as dependency for the next)
    cudaGraphNode_t prev_phase3_up_left_node,   prev_phase3_up_right_node, 
                    prev_phase3_down_left_node, prev_phase3_down_right_node;
         
    for(int t = 0; t < num_rounds; t++) {
        
        // variables needed as function phases args
        int zero = 0;
        int next_t = t+1;

        // --------------------------------------------------------------------------------------------

        // START PHASE 1 NODE & DEPENDENCIES DEFINITION

        // phase 1: self-dependent block

        // execute_round_device_v_4_0_phase_1<<<
        //     num_blocks, 
        //     threads_per_block, 
        //     ARR_MATRIX_SIZE_BANK_CONFICT(B, bank_conflict_phase_1)*sizeof(int), 
        //     graph_stream>>>(dev_matrix, n, t, bank_conflict_phase_1);

        // HANDLE_ERROR(cudaDeviceSynchronize());

        // clear temp dependencies vector
        nodeDependencies.clear(); 

        // phase 1 node of block (t,t)
        cudaGraphNode_t phase1_node;
        // function parameters
        void* phase1_args[4] = { (void*) &dev_matrix, (void*) &n, (void*) &t, (void*) &bank_conflict_phase_1 };
        // node parameters
        cudaKernelNodeParams phase1_params;
        phase1_params.func = (void*) execute_round_device_v_4_0_phase_1;
        phase1_params.gridDim = num_blocks;
        phase1_params.blockDim = threads_per_block;
        phase1_params.sharedMemBytes = max(
            ARR_MATRIX_SIZE_BANK_CONFICT(B, bank_conflict_phase_1)*sizeof(int), 
            2*B*B*sizeof(int)
        );
        phase1_params.sharedMemBytes = ARR_MATRIX_SIZE_BANK_CONFICT(B, bank_conflict_phase_1)*sizeof(int);
        phase1_params.kernelParams = (void**) phase1_args;
        phase1_params.extra = NULL;
        // assign dependencies of phase 1
        if (t > 0) {
            // round after first should depend from previous
            nodeDependencies.push_back(prev_phase3_up_left_node);
            nodeDependencies.push_back(prev_phase3_up_right_node);
            nodeDependencies.push_back(prev_phase3_down_right_node);
            nodeDependencies.push_back(prev_phase3_down_left_node);
        } else {
            // first round depends 
            nodeDependencies.push_back(copy_host_to_dev_node);
        }
        // add node to graph with its dependencies
        HANDLE_ERROR(cudaGraphAddKernelNode(
            &phase1_node, graph, 
            nodeDependencies.data(), nodeDependencies.size(), 
            &phase1_params
        ));
        
        // END PHASE 1 NODE & DEPENDENCIES DEFINITION

        // -------------------------------------------------------------------------------------------_____
        
        // START PHASE 2 NODE & DEPENDENCIES DEFINITION

        // phase 2: row and cols
        // all blocks that share a row or a column with the self dependent, so
        //  -   all blocks just above or under t
        //  -   all block at left and at right of t

        // clear temp dependencies vector
        nodeDependencies.clear();

        // assign dependencies of phase 2 for all zones (up, down, right, left)
        nodeDependencies.push_back(phase1_node);

        // --------------- UP ZONE 
        // execute_round_device_v_4_0_phase_2_col_portion<<<
        //     num_blocks, threads_per_block, 
        //     2*B*B*sizeof(int), 
        //     graph_stream>>>(dev_matrix, n, t, 0, t);

        // phase 2 up: node of blocks (i,t) with i < t
        cudaGraphNode_t phase2_up_node;
        // function parameters, shared also with left zone
        void* phase2_up_left_args[5] = { (void*) &dev_matrix, &n, &t, &zero, &t };
        // node parameters (copy from phase 1 and edit the different ones)
        cudaKernelNodeParams phase2_up_params = cuda_graph_node_params_copy(phase1_params);
        phase2_up_params.func = (void*) execute_round_device_v_4_0_phase_2_col_portion;
        phase2_up_params.sharedMemBytes = 2*B*B*sizeof(int);
        phase2_up_params.kernelParams = (void**) phase2_up_left_args;
        // add node to graph with its dependencies
        HANDLE_ERROR(cudaGraphAddKernelNode(
            &phase2_up_node, graph, 
            nodeDependencies.data(), nodeDependencies.size(), 
            &phase2_up_params
        ));

        // ----------------- LEFT ZONE
        // execute_round_device_v_4_0_phase_2_row_portion<<<
        //     num_blocks, threads_per_block, 
        //     2*B*B*sizeof(int), 
        //     graph_stream>>>(dev_matrix, n, t, 0, t);

        // phase 2 left: node of blocks (t,j) with j < t
        cudaGraphNode_t phase2_left_node;
        // function parameters are reused from up: phase2_up_left_args
        // (all as up zone)
        // node parameters (same as phase 2 UP, except for the function pointer)
        cudaKernelNodeParams phase2_left_params = cuda_graph_node_params_copy(phase2_up_params);
        phase2_left_params.func = (void*) execute_round_device_v_4_0_phase_2_row_portion;
        // add node to graph with its dependencies
        HANDLE_ERROR(cudaGraphAddKernelNode(
            &phase2_left_node, graph, 
            nodeDependencies.data(), nodeDependencies.size(), 
            &phase2_left_params
        ));

        // ------------------ DOWN ZONE
        // execute_round_device_v_4_0_phase_2_col_portion<<<
        //     num_blocks, threads_per_block, 
        //     2*B*B*sizeof(int), 
        //     graph_stream>>>(dev_matrix, n, t, t+1, num_blocks);

        // phase 2 down: node of blocks (i,t) with i > t
        cudaGraphNode_t phase2_down_node;
        // function parameters
        void* phase2_down_right_args[5] = { (void*) &dev_matrix, &n, &t, &next_t, &num_rounds};
        // node parameters (same as phase 2 UP, except for function parameters)
        cudaKernelNodeParams phase2_down_params = cuda_graph_node_params_copy(phase2_up_params);
        phase2_down_params.kernelParams = (void**) phase2_down_right_args;
        // add node to graph with its dependencies
        HANDLE_ERROR(cudaGraphAddKernelNode(
            &phase2_down_node, graph, 
            nodeDependencies.data(), nodeDependencies.size(), 
            &phase2_down_params
        ));

        // ----------------- RIGHT ZONE
        // execute_round_device_v_4_0_phase_2_row_portion<<<
        //     num_blocks, threads_per_block, 
        //     2*B*B*sizeof(int), 
        //     graph_stream>>>(dev_matrix, n, t, t+1, num_blocks);
        
        // phase 2 right: node of blocks (t,j) with j > t
        cudaGraphNode_t phase2_right_node;
        // function parameters are reused from down: phase2_down_right_args
        // (all as down zone)
        // node parameters (same as phase 2 DOWN, except for function pointer)
        cudaKernelNodeParams phase2_right_params = cuda_graph_node_params_copy(phase2_down_params);
        phase2_right_params.func = (void*) execute_round_device_v_4_0_phase_2_row_portion;
        // add node to graph with its dependencies
        HANDLE_ERROR(cudaGraphAddKernelNode(
            &phase2_right_node, graph, 
            nodeDependencies.data(), nodeDependencies.size(), 
            &phase2_right_params
        ));

        // END PHASE 2 NODE & DEPENDENCIES DEFINITION

        // -------------------------------------------------------------------------

        // START PHASE 3 NODE & DEPENDENCIES DEFINITION
        
        // phase 3: all the remaining blocks, so all the blocks that don't share a row or a col with t
        
        // UP-LEFT ZONE
        // execute_round_device_v_4_0_phase_3_portion<<<
        //     num_blocks, threads_per_block, 
        //     2*B*B*sizeof(int), 
        //     graph_stream>>>(dev_matrix, n, t, 0, 0, t, t);

        // clear temp dependencies vector
        nodeDependencies.clear();
        // assign dependencies of phase 3 only for up-left zone
        nodeDependencies.push_back(phase2_up_node);
        nodeDependencies.push_back(phase2_left_node);

        // phase 3 up-left: node of blocks (i,j) with i < t and j < t
        cudaGraphNode_t phase3_up_left_node;
        // function parameters
        void* phase3_up_left_args[7] = { (void*) &dev_matrix, &n, &t, &zero, &zero, &t, &t };
        // node parameters (function pointer is shared for all phase 3 zones)
        // NB: copy from phase 2 and edit the different ones
        cudaKernelNodeParams phase3_up_left_params = cuda_graph_node_params_copy(phase2_up_params);
        phase3_up_left_params.func = (void*) execute_round_device_v_4_0_phase_3_portion;
        phase3_up_left_params.kernelParams = (void**) phase3_up_left_args;
        // add node to graph with its dependencies
        HANDLE_ERROR(cudaGraphAddKernelNode(
            &phase3_up_left_node, graph, 
            nodeDependencies.data(), nodeDependencies.size(), 
            &phase3_up_left_params
        ));

        // UP-RIGHT ZONE
        // execute_round_device_v_4_0_phase_3_portion<<<
        //     num_blocks, threads_per_block, 
        //     2*B*B*sizeof(int), 
        //     graph_stream>>>(dev_matrix, n, t, 0, t+1, t, num_rounds);
        
        // clear temp dependencies vector
        nodeDependencies.clear();
        // assign dependencies of phase 3 only for up-right zone
        nodeDependencies.push_back(phase2_up_node);
        nodeDependencies.push_back(phase2_right_node);

        // phase 3 up-right: node of blocks (i,j) with i < t and j > t
        cudaGraphNode_t phase3_up_right_node;
        // function parameters 
        void* phase3_up_right_args[7] = {(void*) &dev_matrix, &n, &t, &zero, &next_t, &t, &num_rounds};
        // node parameters (function pointer is shared for all phase 3 zones)
        cudaKernelNodeParams phase3_up_right_params = cuda_graph_node_params_copy(phase3_up_left_params);
        phase3_up_right_params.kernelParams = (void**) phase3_up_right_args;
        // add node to graph with its dependencies
        HANDLE_ERROR(cudaGraphAddKernelNode(
            &phase3_up_right_node, graph, 
            nodeDependencies.data(), nodeDependencies.size(), 
            &phase3_up_right_params
        ));

        // DOWN-RIGHT ZONE
        // execute_round_device_v_4_0_phase_3_portion<<<
        //     num_blocks, threads_per_block, 
        //     2*B*B*sizeof(int), 
        //     graph_stream>>>(dev_matrix, n, t, t+1, t+1, num_rounds, num_rounds);

        // clear temp dependencies vector
        nodeDependencies.clear();
        // assign dependencies of phase 3 only for down-right zone
        nodeDependencies.push_back(phase2_down_node);
        nodeDependencies.push_back(phase2_right_node);

        // phase 3 down-right: node of blocks (i,j) with i > t and j > t
        cudaGraphNode_t phase3_down_right_node;
        // function parameters 
        void* phase3_down_right_args[7] = {(void*) &dev_matrix, &n, &t, &next_t, &next_t, &num_rounds, &num_rounds};
        // node parameters (function pointer is shared for all phase 3 zones)
        cudaKernelNodeParams phase3_down_right_params = cuda_graph_node_params_copy(phase3_up_left_params);
        phase3_down_right_params.kernelParams = (void**) phase3_down_right_args;
        // add node to graph with its dependencies
        HANDLE_ERROR(cudaGraphAddKernelNode(
            &phase3_down_right_node, graph, 
            nodeDependencies.data(), nodeDependencies.size(), 
            &phase3_down_right_params
        ));

        // DOWN-LEFT ZONE
        // execute_round_device_v_4_0_phase_3_portion<<<
        //     num_blocks, threads_per_block, 
        //     2*B*B*sizeof(int), 
        //     graph_stream>>>(dev_matrix, n, t, t+1, 0, num_rounds, t);
        
        // clear temp dependencies vector
        nodeDependencies.clear();
        // assign dependencies of phase 3 only for down-left zone
        nodeDependencies.push_back(phase2_down_node);
        nodeDependencies.push_back(phase2_left_node);

        // phase 3 down-left: node of blocks (i,j) with i > t and j < t
        cudaGraphNode_t phase3_down_left_node;
        // function parameters 
        void* phase3_down_left_args[7] = {(void*) &dev_matrix, &n, &t, &next_t, &zero, &num_rounds, &t};
        // node parameters (function pointer is shared for all phase 3 zones)
        cudaKernelNodeParams phase3_down_left_params = cuda_graph_node_params_copy(phase3_up_left_params);
        phase3_down_left_params.kernelParams = (void**) phase3_down_left_args;
        // add node to graph with its dependencies
        HANDLE_ERROR(cudaGraphAddKernelNode(
            &phase3_down_left_node, graph, 
            nodeDependencies.data(), nodeDependencies.size(), 
            &phase3_down_left_params
        ));

        // END PHASE 3 NODE & DEPENDENCIES DEFINITION

        // -------------------------------------------------------------------------------

        // save phase 3 nodes to previous nodes
        prev_phase3_up_left_node    = phase3_up_left_node;
        prev_phase3_up_right_node   = phase3_up_right_node;
        prev_phase3_down_right_node = phase3_down_right_node;
        prev_phase3_down_left_node  = phase3_down_left_node;
    }

    // -------------------------------------------------------------------------------------

    // START MEMCPY DEVICE->HOST NODE & DEPENDENCIES DEFINITION

    // last node: end with memcpy device -> host
    cudaGraphNode_t copy_dev_to_host_node;
    // node parameters
    cudaMemcpy3DParms copy_dev_to_host_params = {0};
    copy_dev_to_host_params.srcArray = NULL;
    copy_dev_to_host_params.srcPos = make_cudaPos(0, 0, 0);
    copy_dev_to_host_params.srcPtr = make_cudaPitchedPtr((void*) dev_matrix, n*sizeof(int), n, n);
    copy_dev_to_host_params.dstArray = NULL;
    copy_dev_to_host_params.dstPos = make_cudaPos(0, 0, 0);
    copy_dev_to_host_params.dstPtr = make_cudaPitchedPtr((void*) matrix, n*sizeof(int), n, n);
    copy_dev_to_host_params.extent = make_cudaExtent(n*sizeof(int), n, 1);
    copy_dev_to_host_params.kind = cudaMemcpyDeviceToHost;

    // clear temp dependencies vector
    nodeDependencies.clear();
    // assign dependencies for final memcpy (all phase 3 zones)
    nodeDependencies.push_back(prev_phase3_up_left_node);
    nodeDependencies.push_back(prev_phase3_up_right_node);
    nodeDependencies.push_back(prev_phase3_down_right_node);
    nodeDependencies.push_back(prev_phase3_down_left_node);

    // add node to graph with its dependencies
    HANDLE_ERROR(cudaGraphAddMemcpyNode(
        &copy_dev_to_host_node, graph, 
        nodeDependencies.data(), nodeDependencies.size(), 
        &copy_dev_to_host_params
    ));
    
    // END MEMCPY DEVICE->HOST NODE & DEPENDENCIES DEFINITION

    // END GRAPH DEFINITION

    // ------------------------------------------------------------------------------------------

    // START GRAPH EXECUTION

    // stream used for executing graph
    cudaStream_t graph_stream;
    cudaStreamCreate(&graph_stream);

    // Instanciate and run graph
    cudaGraphExec_t instance;
    HANDLE_ERROR(cudaGraphInstantiate(&instance, graph, NULL, NULL, 0));
    HANDLE_ERROR(cudaGraphLaunch(instance, graph_stream));
    HANDLE_ERROR(cudaStreamSynchronize(graph_stream));
    
    // END GRAPH EXECUTION
    
    // ------------------------------------------------------------------------------------------

    // START MEMORY CLEANING

    // destroy graph execution instance
    HANDLE_ERROR(cudaGraphExecDestroy(instance));
    // destroy graph structure
    HANDLE_ERROR(cudaGraphDestroy(graph));

    // free device matrix
    HANDLE_ERROR(cudaFree(dev_matrix));

    // destroy streams
    HANDLE_ERROR(cudaStreamDestroy(graph_stream));

    // END MEMORY CLEANING

}


__global__ void execute_round_device_v_4_0_phase_1(int *matrix, int n, int t, bool handle_bank_conflict) {

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


__global__ void execute_round_device_v_4_0_phase_2_row_portion(int *matrix, int n, int t, int start_col, int end_col) {
    
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


__global__ void execute_round_device_v_4_0_phase_2_col_portion(int *matrix, int n, int t, int start_row, int end_row) {
    
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



__global__ void execute_round_device_v_4_0_phase_3_portion(int *matrix, int n, int t, int start_row, int start_col, int end_row, int end_col) {

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
