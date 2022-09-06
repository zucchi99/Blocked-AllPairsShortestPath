
#include "include/include_needed_libraries.cuh"

//main device code
void floyd_warshall_blocked_device_v_1_0(int *matrix, int n, int B);

//rounds code
__global__ void execute_round_device_v_1_0(int *matrix, int n, int t, int row, int col, int B);

int main(int argc, char *argv[]) {

    return handle_arguments_and_execute(argc, argv, (void(*) (int*, int, int)) &floyd_warshall_blocked_device_v_1_0);

}

__global__ void execute_round_device_v_1_0(int *matrix, int n, int t, int row, int col, int B) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    int i = tid/n;  // row
    int j = tid%n;  // col

    //foreach k: t*B <= t < t+B
    for (int k = t * B; k < (t+1) * B; k++) {

        int a, b;
        bool run_this = ((i >= row*B) && (i < (row+1)*B) && (j >= col*B) && (j < (col+1)*B));

        // check if thread correspond to one of the cells in current block
        if (run_this) {

            // WARNING: do NOT put the macro directly into 
            a = matrix[i*n + j];
            b = sum_if_not_infinite(matrix[i*n + k], matrix[k*n + j], INF); 
        }

        __syncthreads();


        if (run_this) {
            matrix[i*n + j] = mmin(a, b);
        }
        
        __syncthreads();

    }
}


void floyd_warshall_blocked_device_v_1_0(int *matrix, int n, int B) {

    int *dev_rand_matrix;
    HANDLE_ERROR(cudaMalloc( (void**) &dev_rand_matrix, n * n* sizeof(int)));
    HANDLE_ERROR(cudaMemcpy(dev_rand_matrix, matrix, n*n*sizeof(int), cudaMemcpyHostToDevice));
    
    int num_rounds = n/B;

    int num_blocks = num_rounds*num_rounds;
    int thread_per_block = B*B; 
    

    for(int t = 0; t < num_rounds; t++) { 

        //arr_execute_round(int *matrix, int n, int t, int row, int col, int B)

        //phase 1: self-dependent block
        execute_round_device_v_1_0<<<num_blocks, thread_per_block>>>(dev_rand_matrix, n, t, t, t, B);
        HANDLE_ERROR(cudaDeviceSynchronize());

        //phase 2 blocks left
        for (int j = t-1; j >= 0; j--) {
            execute_round_device_v_1_0<<<num_blocks, thread_per_block>>>(dev_rand_matrix, n, t, t, j, B);
            // HANDLE_ERROR(cudaDeviceSynchronize());  
        }

        //phase 2 blocks above
        for (int i = t-1; i >= 0; i--) {
            execute_round_device_v_1_0<<<num_blocks, thread_per_block>>>(dev_rand_matrix, n, t, i, t, B);
            // HANDLE_ERROR(cudaDeviceSynchronize());  
        }

        //phase 2 blocks below
        for (int i = t+1; i < num_rounds; i++) {
            execute_round_device_v_1_0<<<num_blocks, thread_per_block>>>(dev_rand_matrix, n, t, i, t, B);
            // HANDLE_ERROR(cudaDeviceSynchronize());  
        }

        //phase 2 blocks right
        for (int j = t+1; j < num_rounds; j++) {
            execute_round_device_v_1_0<<<num_blocks, thread_per_block>>>(dev_rand_matrix, n, t, t, j, B);
            // HANDLE_ERROR(cudaDeviceSynchronize());  
        }

        HANDLE_ERROR(cudaDeviceSynchronize());
        
        //phase 3 blocks above and right
        for (int j = t+1; j < num_rounds; j++) {
            for (int i = t-1; i >= 0; i--) {
                execute_round_device_v_1_0<<<num_blocks, thread_per_block>>>(dev_rand_matrix, n, t, i, j, B);
                // HANDLE_ERROR(cudaDeviceSynchronize());  
            }
        }
        //phase 3 blocks above and left
        for (int j = t-1; j >= 0; j--) {
            for (int i = t-1; i >= 0; i--) {
                execute_round_device_v_1_0<<<num_blocks, thread_per_block>>>(dev_rand_matrix, n, t, i, j, B);
                // HANDLE_ERROR(cudaDeviceSynchronize());  
            }
        }
        //phase 3 blocks below and left
        for (int j = t-1; j >= 0; j--) {
            for (int i = t+1; i < num_rounds; i++) {
                execute_round_device_v_1_0<<<num_blocks, thread_per_block>>>(dev_rand_matrix, n, t, i, j, B);
                // HANDLE_ERROR(cudaDeviceSynchronize());  
            }
        }      
        //phase 3 blocks below and right
        for (int j = t+1; j < num_rounds; j++) {
            for (int i = t+1; i < num_rounds; i++) {
                execute_round_device_v_1_0<<<num_blocks, thread_per_block>>>(dev_rand_matrix, n, t, i, j, B);
                // HANDLE_ERROR(cudaDeviceSynchronize());  
            }
        }

        // HANDLE_ERROR(cudaDeviceSynchronize());   
    }

    HANDLE_ERROR(cudaDeviceSynchronize());  

    HANDLE_ERROR(cudaMemcpy(matrix, dev_rand_matrix, n*n*sizeof(int), cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaFree(dev_rand_matrix));
}

