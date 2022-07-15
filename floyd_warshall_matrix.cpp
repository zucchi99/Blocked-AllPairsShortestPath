
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>

#include "include/adj_matrix_utils.hpp"
// #include "src/adj_matrix_utils.cpp"

#define min(a,b) ((a < b) ? a : b)

void floyd_warshall(int **matrix, int n);
void floyd_warshall_blocked(int **matrix, int n, int B);
int sum_if_not_infinite(int a, int b, int infinity);
void execute_round(int **matrix, int n, int t, int row, int col, int B);

int main() {

    //matrix size n*n
    size_t n = 6;

    //if no weights in graph:
    //int INF = (n * (n-1) / 2) + 1;

    int BLOCKING_FACTOR = 2;

    //memory allocation 
    // int **rand_matrix_1 = (int **) malloc(sizeof(int *) * n);
    // for (int i = 0; i < n; i++) {
    //     rand_matrix_1[i] = (int *) malloc(sizeof(int) * n);
    // }

    // int **rand_matrix_2 = (int **) malloc(sizeof(int *) * n);
    // for (int i = 0; i < n; i++) {
    //     rand_matrix_2[i] = (int *) malloc(sizeof(int) * n);
    // }

    int **rand_matrix_1;
    int **rand_matrix_2;

    //random seed
    int rand_seed = time(NULL);
    printf("seed: %d\n", rand_seed);
    srand(rand_seed);

    //matrix initialization
    rand_matrix_1 = generate_graph(n, rand_seed);

    //matrix print
    printf("input adjacency matrix %lux%lu:\n", n, n);
    print_matrix(rand_matrix_1, n, n);

    //floyd_warshall execution
    floyd_warshall(rand_matrix_1, n);

    //print floyd_warshall output
    printf("output adjacency matrix classic %lux%lu:\n", n, n);
    print_matrix(rand_matrix_1, n, n);

    //---------------------------------------------------------------

    //matrix initialization with same seed
    rand_matrix_2 = generate_graph(n, rand_seed);
    
    //floyd_warshall_blocked execution
    floyd_warshall_blocked(rand_matrix_2, n, BLOCKING_FACTOR);
    
    //print floyd_warshall_blocked output
    printf("output adjacency matrix blocked %lux%lu:\n", n, n);
    print_matrix(rand_matrix_2, n, n);

    //---------------------------------------------------------------

    //compare matrixes output
    bool are_the_same = same_matrix(rand_matrix_1, rand_matrix_2, n, n);
    printf("Matrixes are equal? %s\n", bool_to_string(are_the_same));

    return 0;
}

void floyd_warshall(int **matrix, int n) {
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

void floyd_warshall_blocked(int **matrix, int n, int B) {

    int num_rounds = n/B;

    for(int t = 0; t < num_rounds; t++) { 

        //execute_round(int **matrix, int n, int t, int row, int col, int B)

        //phase 1: self-dependent block
        execute_round(matrix, n, t, t, t, B);

        //phase 2 blocks left
        for (int j = t-1; j >= 0; j--) {
            execute_round(matrix, n, t, t, j, B);
        }

        //phase 2 blocks above
        for (int i = t-1; i >= 0; i--) {
            execute_round(matrix, n, t, i, t, B);
        }

        //phase 2 blocks below
        for (int i = t+1; i < num_rounds; i++) {
            execute_round(matrix, n, t, i, t, B);
        }

        //phase 2 blocks right
        for (int j = t+1; j < num_rounds; j++) {
            execute_round(matrix, n, t, t, j, B);
        }
        
        //phase 2,3: remaining blocks
        //phase 3 blocks above and right
        for (int j = t+1; j < num_rounds; j++) {
            for (int i = t-1; i >= 0; i--) {
                execute_round(matrix, n, t, i, j, B);
            }
        }
        //phase 3 blocks above and left
        for (int j = t-1; j >= 0; j--) {
            for (int i = t-1; i >= 0; i--) {
                execute_round(matrix, n, t, i, j, B);
            }
        }
        //phase 3 blocks below and left
        for (int j = t-1; j >= 0; j--) {
            for (int i = t+1; i < num_rounds; i++) {
                execute_round(matrix, n, t, i, j, B);
            }
        }      
        //phase 3 blocks below and right
        for (int j = t+1; j < num_rounds; j++) {
            for (int i = t+1; i < num_rounds; i++) {
                execute_round(matrix, n, t, i, j, B);
            }
        }   
        
    }
}

void execute_round(int **matrix, int n, int t, int row, int col, int B) {
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

int sum_if_not_infinite(int a, int b, int infinity) {
    bool isInf = (a == infinity) || (b == infinity);
    return isInf ? infinity : a+b;
}