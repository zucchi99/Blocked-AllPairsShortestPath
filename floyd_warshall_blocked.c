
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>

#define INF __INT16_MAX__
#define min(a,b) ((a < b) ? a : b)

void print_array(int *array, int size);
void print_matrix(int **matrix, int m, int n);
void floyd_warshall_blocked(int **matrix, int n, int B);
int sum_if_not_infinite(int a, int b, int infinity);

#define DENSITY 55 //%
#define MIN_COST 1
#define MAX_COST 20

int main() {

    //matrix size n*n
    size_t n = 4;
    //int INF = (n * (n-1) / 2) + 1;

    int BLOCKING_FACTOR = 2;

    //memory allocation 
    int **rand_matrix = (int **) malloc(sizeof(int *) * n);
    for (int i = 0; i < n; i++) {
        rand_matrix[i] = (int *) malloc(sizeof(int) * n);
    }

    //random seed
    //srand(time(NULL));
    srand(10);

    //matrix initialization
    for (int i = 0; i < n; i++) {
        for (int j = i+1; j < n; j++) {
            bool add_edge = (rand() % 100) <= DENSITY;
            int val = (rand() % MAX_COST) + MIN_COST;
            rand_matrix[i][j] = add_edge ? val : INF;
            //non-oriented graph
            rand_matrix[j][i] = rand_matrix[i][j];
        }
    }

    printf("input adjacency matrix %lux%lu:\n", n, n);
    print_matrix(rand_matrix, n, n);

    floyd_warshall_blocked(rand_matrix, n, BLOCKING_FACTOR);

    printf("output adjacency matrix %lux%lu:\n", n, n);
    print_matrix(rand_matrix, n, n);

    return 0;
}

void floyd_warshall_blocked(int **matrix, int n, int B) {
    int num_rounds = n/B;
    for(int t = 0; t < num_rounds; t++) { 

        //phase 1: self-dependent block
        //foreach k: t*B <= t < t+B
        for (int k = t * B; k < (t+1) * B; k++) {
            //foreach i,j in the self-dependent block
            for (int i = t*B; i < (t+1)*B; i++) {
                for (int j = t*B; j < (t+1)*B; j++) {
                    int a = matrix[i][j];
                    int b = sum_if_not_infinite(matrix[i][k], matrix[k][j], INF);
                    matrix[i][j] = min(a, b);
                }
            }
        }

        //phase 2,3: remaining blocks
        //foreach k: t*B <= t < t+B
        for (int k = t * B; k < (t+1) * B; k++) {
            //foreach i,j in the self-dependent block
            for (int i = t*B; i < (t+1)*B; i++) {
                for (int j = t*B; j < (t+1)*B; j++) {
                    int a = matrix[i][j];
                    int b = sum_if_not_infinite(matrix[i][k], matrix[k][j], INF);
                    matrix[i][j] = min(a, b);
                }
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