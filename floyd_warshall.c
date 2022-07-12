
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>

#define INF __INT16_MAX__

#define DENSITY 60 //%
#define MIN_COST 1
#define MAX_COST 20

#define min(a,b) ((a < b) ? a : b)

void print_array(int *array, int size);
void print_matrix(int **matrix, int m, int n);
void floyd_warshall(int **matrix, int n);
void floyd_warshall_blocked(int **matrix, int n, int B);
int sum_if_not_infinite(int a, int b, int infinity);
void execute_round(int **matrix, int n, int t, int row, int col, int B);
void generate_graph(int **matrix, int n, int seed);

int main() {

    //matrix size n*n
    size_t n = 6;

    //if no weights in graph:
    //int INF = (n * (n-1) / 2) + 1;

    int BLOCKING_FACTOR = 2;

    //memory allocation 
    int **rand_matrix = (int **) malloc(sizeof(int *) * n);
    for (int i = 0; i < n; i++) {
        rand_matrix[i] = (int *) malloc(sizeof(int) * n);
    }

    //random seed
    int rand_seed = time(NULL);
    srand(rand_seed);

    //matrix initialization
    generate_graph(rand_matrix, n, rand_seed);
    printf("input adjacency matrix %lux%lu:\n", n, n);
    print_matrix(rand_matrix, n, n);
    floyd_warshall(rand_matrix, n);
    printf("output adjacency matrix classic %lux%lu:\n", n, n);
    print_matrix(rand_matrix, n, n);

    //matrix initialization
    generate_graph(rand_matrix, n, rand_seed);
    //printf("input adjacency matrix %lux%lu:\n", n, n);
    //print_matrix(rand_matrix, n, n);
    floyd_warshall_blocked(rand_matrix, n, BLOCKING_FACTOR);
    printf("output adjacency matrix blocked %lux%lu:\n", n, n);
    print_matrix(rand_matrix, n, n);

    return 0;
}


void generate_graph(int **matrix, int n, int seed) {
    srand(seed);
    for (int i = 0; i < n; i++) {
        matrix[i][i] = 0;
        for (int j = i+1; j < n; j++) {
            bool add_edge = (rand() % 100) <= DENSITY;
            int val = (rand() % MAX_COST) + MIN_COST;
            matrix[i][j] = add_edge ? val : INF;
            //non-oriented graph
            matrix[j][i] = matrix[i][j];
        }
    }
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

        //phase 2,3: remaining blocks
        //phase 2 blocks right
        for (int j = t+1; j < num_rounds; j++) {
            execute_round(matrix, n, t, t, j, B);
        }
        //phase 2 blocks above
        for (int i = t-1; i >= 0; i--) {
            execute_round(matrix, n, t, i, t, B);
        }
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

        //phase 2 blocks left
        for (int j = t-1; j >= 0; j--) {
            execute_round(matrix, n, t, t, j, B);
        }
        //phase 2 blocks below
        for (int i = t+1; i < num_rounds; i++) {
            execute_round(matrix, n, t, i, t, B);
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

void print_matrix(int **matrix, int m, int n) {
    printf("[\n");
    for (int i = 0; i < m; i++) {
        printf(" ");
        print_array(matrix[i], n);

    }
    printf("]\n");
}

void print_array(int *array, int size) {
    printf("[");
    for (int i = 0; i < size; i++) {
        if (array[i] < INF)
            printf("%d", array[i]);
        else 
            printf("-");
        if (i < size-1) printf(", ");
    }
    printf("]\n");
}