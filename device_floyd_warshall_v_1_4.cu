#include "include/include_needed_libraries.cuh"

//main device code
void floyd_warshall_blocked_device_v_1_4(int *matrix, int n, int B);

//rounds code
__global__ void execute_round_device_v_1_4_phase_1(int *matrix, int n, int t, int B);
__global__ void execute_round_device_v_1_4_phase_2_row(int *matrix, int n, int t, int B);
__global__ void execute_round_device_v_1_4_phase_2_col(int *matrix, int n, int t, int B);
__global__ void execute_round_device_v_1_4_phase_3(int *matrix, int n, int t, int B);

int main(int argc, char *argv[]) {

    return handle_arguments_and_execute(argc, argv, (void(*) (int*, int, int)) &floyd_warshall_blocked_device_v_1_4);

}

void floyd_warshall_blocked_device_v_1_4(int *matrix, int n, int B) {

    assert(n%B == 0);                       // B must divide n
    assert(B*B<=MAX_BLOCK_SIZE);            // B*B cannot exceed mmax block size

    int *dev_rand_matrix;
    HANDLE_ERROR(cudaMalloc( (void**) &dev_rand_matrix, n * n* sizeof(int)));
    HANDLE_ERROR(cudaMemcpy(dev_rand_matrix, matrix, n*n*sizeof(int), cudaMemcpyHostToDevice));

    int num_rounds = n/B;
     
    for(int t = 0; t < num_rounds; t++) { 

        //arr_execute_round(int *matrix, int n, int t, int row, int col, int B)

        //phase 1: self-dependent block
        dim3 num_blocks_phase_1(1, 1);
        dim3 threads_per_block_phase_1(B, B);

        execute_round_device_v_1_4_phase_1<<<num_blocks_phase_1, threads_per_block_phase_1>>>(dev_rand_matrix, n, t, B);
        HANDLE_ERROR(cudaDeviceSynchronize());

        // phase 2: all blocks that share a row or a column with the self dependent, so
        //  -   all blocks just above or under t
        //  -   all block at left and at right of t

        dim3 num_blocks_phase_2(1, num_rounds-1);  

        execute_round_device_v_1_4_phase_2_row<<<num_blocks_phase_2, threads_per_block_phase_1>>>(dev_rand_matrix, n, t, B);
        execute_round_device_v_1_4_phase_2_col<<<num_blocks_phase_2, threads_per_block_phase_1>>>(dev_rand_matrix, n, t, B);

        HANDLE_ERROR(cudaDeviceSynchronize());

        // phase 3: all the remaining blocks, so all the blocks that don't share a row or a col with t

        dim3 num_blocks_phase_3(num_rounds-1, num_rounds-1); 

        execute_round_device_v_1_4_phase_3<<<num_blocks_phase_3, threads_per_block_phase_1>>>(dev_rand_matrix, n, t, B);
        HANDLE_ERROR(cudaDeviceSynchronize()); 
    }

    // HANDLE_ERROR(cudaDeviceSynchronize());  

    HANDLE_ERROR(cudaMemcpy(matrix, dev_rand_matrix, n*n*sizeof(int), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaFree(dev_rand_matrix));
}


__global__ void execute_round_device_v_1_4_phase_1(int *matrix, int n, int t, int B) {

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

    //foreach k: t*B <= t < t+B
    for (int k = BLOCK_START(t,B); k < BLOCK_END(t,B); k++) {

        int b = sum_if_not_infinite(matrix[i*n + k], matrix[k*n + j], INF); 

        if (b < matrix[i*n + j]) {
            matrix[i*n + j] = b;
        }
        
        __syncthreads();
    }
}

__global__ void execute_round_device_v_1_4_phase_2_row(int *matrix, int n, int t, int B) {

    // Launched blocks and correspondent position in the matrix
    //  -   blockIdx.x says if I am iterating row or cols, 
    //  -   blockIdx.y says something about which row or col)
    //  -   threadIdx.x and threadIdx.y are relative position of cell in block

    //  L1  L2  L3  R1  R2

    //  .   .   .   U1  .   .
    //  .   .   .   U2  .   .
    //  .   .   .   U3  .   .
    //  L1  L2  L3  -   R1  R2
    //  .   .   .   D1  .   .
    //  .   .   .   D2  .   .

    int i, j;

    // it's a row ...
    i = BLOCK_START(t, B) + threadIdx.x;

    if (blockIdx.y < t) {

        // ... and it's the left one
        j = BLOCK_START(blockIdx.y, B) + threadIdx.y;

    } else {
        
        // ... and it's the right one
        j = BLOCK_START(blockIdx.y, B) + B + threadIdx.y;
    }

    //foreach k: t*B <= t < t+B
    for (int k = BLOCK_START(t,B); k < BLOCK_END(t,B); k++) {

        int b = sum_if_not_infinite(matrix[i*n + k], matrix[k*n + j], INF); 

        if (b < matrix[i*n + j]) {
            matrix[i*n + j] = b;
        }

        //printf("i:%d, j:%d, k:%d\n", i, j, k);

        __syncthreads();

    }
}

__global__ void execute_round_device_v_1_4_phase_2_col(int *matrix, int n, int t, int B) {

    // Launched blocks and correspondent position in the matrix
    //  -   blockIdx.x says if I am iterating row or cols, 
    //  -   blockIdx.y says something about which row or col)
    //  -   threadIdx.x and threadIdx.y are relative position of cell in block

    //  U1  U2  U3  D1  D2

    //  .   .   .   U1  .   .
    //  .   .   .   U2  .   .
    //  .   .   .   U3  .   .
    //  L1  L2  L3  -   R1  R2
    //  .   .   .   D1  .   .
    //  .   .   .   D2  .   .

    int i, j;

    // it's a column ...
    j = BLOCK_START(t, B) + threadIdx.y;

    if (blockIdx.y < t) {

        // ... and it's the up one
        i = BLOCK_START(blockIdx.y, B) + threadIdx.x;

    } else {

        // ... and it's the down one
        i = BLOCK_START(blockIdx.y, B) + B + threadIdx.x;
    }

    //foreach k: t*B <= t < t+B
    for (int k = BLOCK_START(t,B); k < BLOCK_END(t,B); k++) {

        int b = sum_if_not_infinite(matrix[i*n + k], matrix[k*n + j], INF); 

        if (b < matrix[i*n + j]) {
            matrix[i*n + j] = b;
        }

        //printf("i:%d, j:%d, k:%d\n", i, j, k);

        __syncthreads();
    }
}


__global__ void execute_round_device_v_1_4_phase_3(int *matrix, int n, int t, int B) {

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

    //foreach k: t*B <= t < t+B
    for (int k = BLOCK_START(t,B); k < BLOCK_END(t,B); k++) {

        int using_k_path = sum_if_not_infinite(matrix[i*n + k], matrix[k*n + j], INF); 

        if (using_k_path < matrix[i*n + j]) {
            matrix[i*n + j] = using_k_path;
        }

        __syncthreads();
    }
}

