#include "include/include_needed_libraries.cuh"
#include "include/lcm.hpp"

#define ARR_MATRIX_INDEX(i,j,n) (i*n+j)
#define ARR_MATRIX_INDEX_TRASP(i,j,n) (i+n*j)

#define SHARED_BANK_N_INT 32
#define ARR_MATRIX_INDEX_BANK_CONFLICT(i, j, n, handle_bank_conflict) (i*n + j + (handle_bank_conflict ? i : 0))
#define ARR_MATRIX_SIZE_BANK_CONFICT(B,handle_bank_conflict) (B*B + (handle_bank_conflict ? (B-1) : 0))

//main device code
void floyd_warshall_blocked_device_v_2_2(int *matrix, int n, int B);

//rounds code
__global__ void execute_round_device_v_2_2_phase_1(int *matrix, int n, int t, bool handle_bank_conflict);
__global__ void execute_round_device_v_2_2_phase_2_row(int *matrix, int n, int t);
__global__ void execute_round_device_v_2_2_phase_2_col(int *matrix, int n, int t);
__global__ void execute_round_device_v_2_2_phase_3(int *matrix, int n, int t);

int main(int argc, char *argv[]) {

    return handle_arguments_and_execute(argc, argv, (void(*) (int*, int, int)) &floyd_warshall_blocked_device_v_2_2);
}

void floyd_warshall_blocked_device_v_2_2(int *matrix, int n, int B) {

    assert(n%B == 0);                       // B must divide n
    assert(B*B<=MAX_BLOCK_SIZE);            // B*B cannot exceed mmax block size

    int *dev_rand_matrix;
    HANDLE_ERROR(cudaMalloc( (void**) &dev_rand_matrix, n*n*sizeof(int)));
    HANDLE_ERROR(cudaMemcpy(dev_rand_matrix, matrix, n*n*sizeof(int), cudaMemcpyHostToDevice));

    int num_rounds = n/B;

    bool bank_conflict_phase_1 = lcm(SHARED_BANK_N_INT, B) <= (B-1)*B;
     
    for(int t = 0; t < num_rounds; t++) { 

        //arr_execute_round(int *matrix, int n, int t, int row, int col, int B)

        //phase 1: self-dependent block
        dim3 num_blocks_phase_1(1, 1);
        dim3 threads_per_block_phase_1(B, B);

        execute_round_device_v_2_2_phase_1<<<
            num_blocks_phase_1, 
            threads_per_block_phase_1, 
            ARR_MATRIX_SIZE_BANK_CONFICT(B, bank_conflict_phase_1)*sizeof(int)
            >>>(dev_rand_matrix, n, t, bank_conflict_phase_1);

        HANDLE_ERROR(cudaDeviceSynchronize());

        // phase 2: all blocks that share a row or a column with the self dependent, so
        //  -   all blocks just above or under t
        //  -   all block at left and at right of t

        // dim3 num_blocks_phase_2(1, num_rounds-1);  

        execute_round_device_v_2_2_phase_2_row<<<num_rounds-1, threads_per_block_phase_1, 2*B*B*sizeof(int)>>>(dev_rand_matrix, n, t);
        execute_round_device_v_2_2_phase_2_col<<<num_rounds-1, threads_per_block_phase_1, 2*B*B*sizeof(int)>>>(dev_rand_matrix, n, t);


        HANDLE_ERROR(cudaDeviceSynchronize());

        // phase 3: all the remaining blocks, so all the blocks that don't share a row or a col with t

        dim3 num_blocks_phase_3(num_rounds-1, num_rounds-1); 

        execute_round_device_v_2_2_phase_3<<<num_blocks_phase_3, threads_per_block_phase_1, 2*B*B*sizeof(int)>>>(dev_rand_matrix, n, t);
        HANDLE_ERROR(cudaDeviceSynchronize()); 
    }

    // HANDLE_ERROR(cudaDeviceSynchronize());  

    HANDLE_ERROR(cudaMemcpy(matrix, dev_rand_matrix, n*n*sizeof(int), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaFree(dev_rand_matrix));
}


__global__ void execute_round_device_v_2_2_phase_1(int *matrix, int n, int t, bool handle_bank_conflict) {

    // Launched block and correspondent position in the matrix

    //  t

    //  .   .   .   .   .   . 
    //  .   .   .   .   .   . 
    //  .   .   .   .   .   . 
    //  .   .   .   t   .   .
    //  .   .   .   .   .   . 
    //  .   .   .   .   .   . 

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

__global__ void execute_round_device_v_2_2_phase_2_row(int *matrix, int n, int t) {

    // Launched block and correspondent position in the matrix

    //  L1  L2  L3  R1  R2      
    //  (trasposed)

    //  .   .   .   U1  .   .
    //  .   .   .   U2  .   .
    //  .   .   .   U3  .   .
    //  L1  L2  L3  -   R1  R2
    //  .   .   .   D1  .   .
    //  .   .   .   D2  .   .

    extern __shared__ int shared_mem[];
    
    int* block_i_j_shared = &shared_mem[0];
    int* block_t_t_shared = &shared_mem[(blockDim.x * blockDim.x)];

    // it's a row ...

    // abs row index 
    int i = BLOCK_START(t, blockDim.x) + threadIdx.x;    
    // abs col index   
    int j = BLOCK_START(blockIdx.x, blockDim.x) + threadIdx.y + ((blockIdx.x >= t) ? blockDim.x : 0); 

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

__global__ void execute_round_device_v_2_2_phase_2_col(int *matrix, int n, int t) {

    // Launched block and correspondent position in the matrix

    //  U1  U2  U3  D1  D2
    //  (trasposed)

    //  .   .   .   U1  .   .
    //  .   .   .   U2  .   .
    //  .   .   .   U3  .   .
    //  L1  L2  L3  -   R1  R2
    //  .   .   .   D1  .   .
    //  .   .   .   D2  .   .

    extern __shared__ int shared_mem[];

    int* block_i_j_shared = &shared_mem[0];
    int* block_t_t_shared = &shared_mem[blockDim.x*blockDim.x];

    // it's a column ...

    // abs row index 
    int i = BLOCK_START(blockIdx.x, blockDim.x) + threadIdx.x + ((blockIdx.x >= t) ? blockDim.x : 0);
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


__global__ void execute_round_device_v_2_2_phase_3(int *matrix, int n, int t) {

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

    extern __shared__ int shared_mem[];

    int* block_i_t_shared = &shared_mem[0];
    int* block_t_j_shared = &shared_mem[blockDim.x*blockDim.x];

    // abs row index
    int i = threadIdx.x + blockIdx.x * blockDim.x + ((blockIdx.x >= t) ? blockDim.x : 0);
    // abs col index
    int j = threadIdx.y + blockIdx.y * blockDim.y + ((blockIdx.y >= t) ? blockDim.x : 0);
    
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

