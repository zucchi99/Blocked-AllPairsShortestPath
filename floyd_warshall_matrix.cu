%%file floyd_warshall.cu

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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
#define bool_to_string(cond) (cond ? "true" : "false")

#define HANDLE_ERROR(err) (handle_error(err, __FILE__, __LINE__))

void print_array(int *array, int size);
void print_matrix(int **matrix, int m, int n);
void floyd_warshall(int **matrix, int n);
void floyd_warshall_blocked(int **matrix, int n, int B);
int sum_if_not_infinite(int a, int b, int infinity);
__global__ void execute_round(int **matrix, int n, int t, int row, int col, int B);
void generate_graph(int **matrix, int n, int seed);
bool same_matrix(int **matrix_1, int **matrix_2, int m, int n);

//error handling
static void handle_error(cudaError_t err, const char *file, int line);
void check_CUDA_error(const char *msg);

int main() {

    //matrix size n*n
    size_t n = 6;

    //if no weights in graph:
    //int INF = (n * (n-1) / 2) + 1;

    int BLOCKING_FACTOR = 2;
    
    //graph generation
    //random seed
    int rand_seed = time(NULL);
    printf("seed: %d\n", rand_seed);
    srand(rand_seed);

    //data definition
    //memory allocation

    //var host con indirizzo host
    //matrix_1 per check con floyd_warshall classico sequenziale
    int **rand_matrix_1 = (int **) malloc(sizeof(int *) * n);
    for (int i = 0; i < n; i++) {
        rand_matrix_1[i] = (int *) malloc(sizeof(int) * n);
    }

    //var host con indirizzo host
    //matrix_2 per input e output col kernel
    int **rand_matrix_2 = (int **) malloc(sizeof(int *) * n);
    for (int i = 0; i < n; i++) {
        rand_matrix_2[i] = (int *) malloc(sizeof(int) * n);
    }

    printf("allocating kernel\n");

    //var host con indirizzo device
    //matrix_2 per esecuzione sul kernel
    int **dev_rand_matrix;
    HANDLE_ERROR(cudaMalloc( (void**) &dev_rand_matrix, n * sizeof(int *)));
    cudaError_t err = cudaMalloc( (void**) &dev_rand_matrix, n * sizeof(int *));
    printf("%s\n", cudaGetErrorString(err));

    printf("allocated list of pointer\n");

    for (int i = 0; i < n; i++) {
        //HANDLE_ERROR(cudaMalloc( (void**) &(dev_rand_matrix[i]), n * sizeof(int)));
        cudaError_t err = cudaMalloc( (void**) &(dev_rand_matrix[i]), n * sizeof(int));
        printf("%s\n", cudaGetErrorString(err));
        printf("%d\n", i);
    }
    
    printf("allocated kernel\n");
 
    //host matrix initialization
    generate_graph(rand_matrix_1, n, rand_seed);
    generate_graph(rand_matrix_2, n, rand_seed);

    //matrix print
    printf("input adjacency matrix %lux%lu:\n", n, n);
    print_matrix(rand_matrix_1, n, n);

    //floyd_warshall execution
    floyd_warshall(rand_matrix_1, n);

    //print floyd_warshall output
    printf("output adjacency matrix classic %lux%lu:\n", n, n);
    print_matrix(rand_matrix_1, n, n);

    //---------------------------------------------------------------
    

    cudaDeviceSynchronize();

    printf("end\n");

    return 0;
    //KERNEL EXECUTION

    //matrix initialization with same seed
    
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

bool same_matrix(int **matrix_1, int **matrix_2, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if(matrix_1[i][j] != matrix_2[i][j]) return false;
        }
    }
    return true;
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

/*
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
*/

__global__ void execute_round(int **matrix, int n, int t, int row, int col, int B) {
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
                //int b = sum_if_not_infinite(matrix[i][k], matrix[k][j], INF);
                bool isInf = (x1 == INF) || (x2 == INF);
                int b = isInf ? INF : x1+x2;
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
            printf("%02d", array[i]);
        else 
            printf("--");
        if (i < size-1) printf(", ");
    }
    printf("]\n");
}

static void handle_error(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
        exit(EXIT_FAILURE);
    }
}

void check_CUDA_error(const char *msg) {
    cudaError_t err = cudaGetLastError();
    if(cudaSuccess != err) {
        fprintf(stderr, "ERRORE CUDA: >%s<: >%s<. Eseguo: EXIT\n", msg, cudaGetErrorString(err) );
        exit(-1);
    }
}