#include "adj_matrix_utils.hpp"

#include <stdio.h>
#include <stdlib.h>

// ---------------------------------------------------------------
//  PRINT UTILS

void print_matrix(int **matrix, int m, int n) {
    printf("[\n");
    for (int i = 0; i < m; i++) {
        printf("  ");
        print_array(matrix[i], n);
    }
    printf("]\n");
}

void print_array(int *array, int size) {
    printf("[");
    for (int i = 0; i < size; i++) {
        print_element(array[i], INF);
        if (i < size-1) printf(", ");
    }
    printf("]\n");
}

void print_element(int val, int infinity) {
    if (val < infinity)
        printf("%02d", val);
    else 
        printf("--");
}

// ---------------------------------------------------------------
// MATRIX GENERATION, COMPARE and others utils

bool same_matrix(int **matrix_1, int **matrix_2, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if(matrix_1[i][j] != matrix_2[i][j]) return false;
        }
    }
    return true;
}

int** generate_graph(int n, int seed) {

    int **matrix = (int **) malloc(sizeof(int *) * n);
    for (int i = 0; i < n; i++) {
        matrix[i] = (int *) malloc(sizeof(int) * n);
    }


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

    return matrix;
}

// ---------------------------------------------------------------
// ARRAY MATRIX FUNCTIONS VARIANTS

void print_arr_matrix(int *matrix, int m, int n) {
    printf("[\n");
    for (int i = 0; i < m; i++) {
        printf("  ");
        print_array(&(matrix[i]), n);
    }
    printf("]\n");
}

bool same_arr_matrix(int *matrix_1, int *matrix_2, int n) {
    for (int i = 0; i < n; i++) {
        if(matrix_1[i] != matrix_2[i]) return false;
    }
    return true;
}

int* generate_arr_graph(int n, int seed) {

    int *matrix = (int *) malloc(sizeof(int *) * n * n);

    srand(seed);
    for (int i = 0; i < n; i++) {
        matrix[i*n + i] = 0;
        for (int j = i+1; j < n; j++) {
            bool add_edge = (rand() % 100) <= DENSITY;
            int val = (rand() % MAX_COST) + MIN_COST;
            matrix[i*n + j] = add_edge ? val : INF;
            //non-oriented graph
            matrix[j*n + i] = matrix[i*n + j];
        }
    }

    return matrix;
}