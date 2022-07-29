#include "../include/host_floyd_warshall.hpp"
#include "../include/adj_matrix_utils.hpp"

// ---------------------------------------------------------------------------
// Matrix data structure version

void host_matrix_floyd_warshall(int **matrix, int n) {
    for(int k = 0; k < n; k++) {
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < n; j++) {
                int a = matrix[i][j];
                int b = sum_if_not_infinite(matrix[i][k], matrix[k][j], INF);
                matrix[i][j] = min(a, b);
            }
        }
    }
}

void host_matrix_floyd_warshall_blocked(int **matrix, int n, int B) {

    int num_rounds = n/B;

    for(int t = 0; t < num_rounds; t++) { 

        //host_matrix_execute_round(int **matrix, int n, int t, int row, int col, int B)

        //phase 1: self-dependent block
        host_matrix_execute_round(matrix, n, t, t, t, B);

        //phase 2 blocks left
        for (int j = t-1; j >= 0; j--) {
            host_matrix_execute_round(matrix, n, t, t, j, B);
        }

        //phase 2 blocks above
        for (int i = t-1; i >= 0; i--) {
            host_matrix_execute_round(matrix, n, t, i, t, B);
        }

        //phase 2 blocks below
        for (int i = t+1; i < num_rounds; i++) {
            host_matrix_execute_round(matrix, n, t, i, t, B);
        }

        //phase 2 blocks right
        for (int j = t+1; j < num_rounds; j++) {
            host_matrix_execute_round(matrix, n, t, t, j, B);
        }
        
        //phase 2,3: remaining blocks
        //phase 3 blocks above and right
        for (int j = t+1; j < num_rounds; j++) {
            for (int i = t-1; i >= 0; i--) {
                host_matrix_execute_round(matrix, n, t, i, j, B);
            }
        }
        //phase 3 blocks above and left
        for (int j = t-1; j >= 0; j--) {
            for (int i = t-1; i >= 0; i--) {
                host_matrix_execute_round(matrix, n, t, i, j, B);
            }
        }
        //phase 3 blocks below and left
        for (int j = t-1; j >= 0; j--) {
            for (int i = t+1; i < num_rounds; i++) {
                host_matrix_execute_round(matrix, n, t, i, j, B);
            }
        }      
        //phase 3 blocks below and right
        for (int j = t+1; j < num_rounds; j++) {
            for (int i = t+1; i < num_rounds; i++) {
                host_matrix_execute_round(matrix, n, t, i, j, B);
            }
        }   
        
    }
}

void host_matrix_execute_round(int **matrix, int n, int t, int row, int col, int B) {

    //foreach k: t*B <= t < t+B
    int block_start = t * B;
    int block_end = (t+1) * B;
    int row_start = row * B;
    int row_end = (row+1) * B;
    int col_start = col * B;
    int col_end = (col+1) * B;

    for (int k = block_start; k < block_end; k++) {
        //foreach i,j in the self-dependent block
        for (int i = row_start; i < row_end; i++) {
            for (int j = col_start; j < col_end; j++) {
                int a = matrix[i][j];
                int x1 = matrix[i][k];
                int x2 =  matrix[k][j];
                int b = sum_if_not_infinite(matrix[i][k], matrix[k][j], INF);
                matrix[i][j] = min(a, b);
                //print_matrix(matrix, n, n);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Array data structure version

void host_array_floyd_warshall(int *matrix, int n) {
    for(int k = 0; k < n; k++) {
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < n; j++) {
                int a = matrix[i*n + j];
                int b = sum_if_not_infinite(matrix[i*n + k], matrix[k*n + j], INF);
                matrix[i*n + j] = min(a, b);
            }
        }
    }
}

void host_array_floyd_warshall_blocked(int *matrix, int n, int B) {

    int num_rounds = n/B;

    for(int t = 0; t < num_rounds; t++) { 

        //host_matrix_execute_round(int *matrix, int n, int t, int row, int col, int B)

        //phase 1: self-dependent block
        host_array_execute_round(matrix, n, t, t, t, B);

        //phase 2 blocks left
        for (int j = t-1; j >= 0; j--) {
            host_array_execute_round(matrix, n, t, t, j, B);
        }

        //phase 2 blocks above
        for (int i = t-1; i >= 0; i--) {
            host_array_execute_round(matrix, n, t, i, t, B);
        }

        //phase 2 blocks below
        for (int i = t+1; i < num_rounds; i++) {
            host_array_execute_round(matrix, n, t, i, t, B);
        }

        //phase 2 blocks right
        for (int j = t+1; j < num_rounds; j++) {
            host_array_execute_round(matrix, n, t, t, j, B);
        }
        
        //phase 2,3: remaining blocks
        //phase 3 blocks above and right
        for (int j = t+1; j < num_rounds; j++) {
            for (int i = t-1; i >= 0; i--) {
                host_array_execute_round(matrix, n, t, i, j, B);
            }
        }
        //phase 3 blocks above and left
        for (int j = t-1; j >= 0; j--) {
            for (int i = t-1; i >= 0; i--) {
                host_array_execute_round(matrix, n, t, i, j, B);
            }
        }
        //phase 3 blocks below and left
        for (int j = t-1; j >= 0; j--) {
            for (int i = t+1; i < num_rounds; i++) {
                host_array_execute_round(matrix, n, t, i, j, B);
            }
        }      
        //phase 3 blocks below and right
        for (int j = t+1; j < num_rounds; j++) {
            for (int i = t+1; i < num_rounds; i++) {
                host_array_execute_round(matrix, n, t, i, j, B);
            }
        }   
        
    }
}

void host_array_execute_round(int *matrix, int n, int t, int row, int col, int B) {
    //foreach k: t*B <= t < t+B
    int block_start = t * B;
    int block_end = (t+1) * B;
    int row_start = row * B;
    int row_end = (row+1) * B;
    int col_start = col * B;
    int col_end = (col+1) * B;
    for (int k = block_start; k < block_end; k++) {
        //foreach i,j in the self-dependent block
        for (int i = row_start; i < row_end; i++) {
            for (int j = col_start; j < col_end; j++) {
                int a = matrix[i*n + j];
                int x1 = matrix[i*n + k];
                int x2 =  matrix[k*n + j];
                int b = sum_if_not_infinite(matrix[i*n + k], matrix[k*n + j], INF);
                matrix[i*n + j] = min(a, b);
                //print_arr_matrix(matrix, n, n);
            }
        }
    }
}