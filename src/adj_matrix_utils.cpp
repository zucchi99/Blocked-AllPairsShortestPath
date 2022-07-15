#include "../include/adj_matrix_utils.h"

#include <stdio.h>
#include <stdlib.h>


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

void print_matrix(int **matrix, int m, int n) {
    printf("[\n");
    for (int i = 0; i < m; i++) {
        printf(" ");
        print_array(matrix[i], n);

    }
    printf("]\n");
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