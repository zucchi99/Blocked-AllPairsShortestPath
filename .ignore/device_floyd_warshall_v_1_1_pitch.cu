#include "../include/include_needed_libraries.cuh"


//main device code
void floyd_warshall_blocked_device_v_1_1_pitch(int *matrix, int n, int B);

//rounds code
__global__ void execute_round_device_v_1_1_pitch(int *matrix, int n, int t, int row, int col, int B, size_t pitch);

int main(int argc, char *argv[]) {

    return handle_arguments_and_execute(argc, argv, (void(*) (int*, int, int)) &floyd_warshall_blocked_device_v_1_1_pitch);

}


void floyd_warshall_blocked_device_v_1_1_pitch(int *matrix, int n, int B) {
    
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
        execute_round_device_v_1_1_pitch<<<num_blocks_phase_1, threads_per_block_phase_1>>>(dev_rand_matrix, n, t, t, t, B, pitch);
        HANDLE_ERROR(cudaDeviceSynchronize());

        //phase 2 blocks left
        for (int j = t-1; j >= 0; j--) {
            // printf("start phase 2 blocks left, t:%d, row:%d, col:%d\n",t,t,j);
            execute_round_device_v_1_1_pitch<<<num_blocks_phase_1, threads_per_block_phase_1>>>(dev_rand_matrix, n, t, t, j, B, pitch);
            // HANDLE_ERROR(cudaDeviceSynchronize());  
        }

        //phase 2 blocks above
        for (int i = t-1; i >= 0; i--) {
            // printf("start phase 2 blocks above, t:%d, row:%d, col:%d\n",t,i,t);
            execute_round_device_v_1_1_pitch<<<num_blocks_phase_1, threads_per_block_phase_1>>>(dev_rand_matrix, n, t, i, t, B, pitch);
            // HANDLE_ERROR(cudaDeviceSynchronize());  
        }

        //phase 2 blocks below
        for (int i = t+1; i < num_rounds; i++) {
            // printf("start phase 2 blocks below, t:%d, row:%d, col:%d\n",t,i,t);
            execute_round_device_v_1_1_pitch<<<num_blocks_phase_1, threads_per_block_phase_1>>>(dev_rand_matrix, n, t, i, t, B, pitch);
            // HANDLE_ERROR(cudaDeviceSynchronize());  
        }

        //phase 2 blocks right
        for (int j = t+1; j < num_rounds; j++) {
            // printf("start phase 2 blocks right, t:%d, row:%d, col:%d\n",t,t,j);
            execute_round_device_v_1_1_pitch<<<num_blocks_phase_1, threads_per_block_phase_1>>>(dev_rand_matrix, n, t, t, j, B, pitch);
            // HANDLE_ERROR(cudaDeviceSynchronize());  
        }

        HANDLE_ERROR(cudaDeviceSynchronize());
        
        //phase 3 blocks above and right
        for (int j = t+1; j < num_rounds; j++) {
            for (int i = t-1; i >= 0; i--) {
                // printf("start phase 3 blocks above and right, t:%d, row:%d, col:%d\n",t,i,j);
                execute_round_device_v_1_1_pitch<<<num_blocks_phase_1, threads_per_block_phase_1>>>(dev_rand_matrix, n, t, i, j, B, pitch);
                // HANDLE_ERROR(cudaDeviceSynchronize());  
            }
        }
        //phase 3 blocks above and left
        for (int j = t-1; j >= 0; j--) {
            for (int i = t-1; i >= 0; i--) {
                // printf("start phase 3 blocks above and left, t:%d, row:%d, col:%d\n",t,i,j);
                execute_round_device_v_1_1_pitch<<<num_blocks_phase_1, threads_per_block_phase_1>>>(dev_rand_matrix, n, t, i, j, B, pitch);
                // HANDLE_ERROR(cudaDeviceSynchronize());  
            }
        }
        //phase 3 blocks below and left
        for (int j = t-1; j >= 0; j--) {
            for (int i = t+1; i < num_rounds; i++) {
                // printf("start phase 3 blocks below and left, t:%d, row:%d, col:%d\n",t,i,j);
                execute_round_device_v_1_1_pitch<<<num_blocks_phase_1, threads_per_block_phase_1>>>(dev_rand_matrix, n, t, i, j, B, pitch);
                // HANDLE_ERROR(cudaDeviceSynchronize());  
            }
        }      
        //phase 3 blocks below and right
        for (int j = t+1; j < num_rounds; j++) {
            for (int i = t+1; i < num_rounds; i++) {
                // printf("start phase 3 blocks below and right, t:%d, row:%d, col:%d\n",t,i,j);
                execute_round_device_v_1_1_pitch<<<num_blocks_phase_1, threads_per_block_phase_1>>>(dev_rand_matrix, n, t, i, j, B, pitch);
                // HANDLE_ERROR(cudaDeviceSynchronize());  
            }
        }

        HANDLE_ERROR(cudaDeviceSynchronize());  
    }

    // HANDLE_ERROR(cudaMemcpy(matrix, dev_rand_matrix, n*n*sizeof(int), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy2D(matrix, width, dev_rand_matrix, pitch, width, height, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaFree(dev_rand_matrix));
}

__global__ void execute_round_device_v_1_1_pitch(int *matrix, int n, int t, int row, int col, int B, size_t pitch) {

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
