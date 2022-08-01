
#include <stdio.h>
#include <stdlib.h>

#include "../include/adj_matrix_utils.hpp"

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

bool same_matrixes(int **matrix_1, int **matrix_2, int m, int n, bool oriented_graph) {
    for (int i = 0; i < m; i++) {
        for (int j = oriented_graph ? 0 : (i+1); j < n; j++) {
            if(matrix_1[i][j] != matrix_2[i][j]) return false;
        }
    }
    return true;
}

int** allocate_matrix(int m, int n) {
    //matrix with row major order:
    //m rows pointers
    int** matrix = (int **) malloc(sizeof(int *) * m);
    for (int i = 0; i < m; i++) {
        //each row has n intengers
        matrix[i] = (int *) malloc(sizeof(int) * n);
    }
    return matrix;
}

void populate_adj_matrix(int **matrix, int n, int seed, bool oriented_graph) {
    for (int i = 0; i < n; i++) {
        //diagonal always zero (distance 0 to myself)
        matrix[i][i] = 0;
        for (int j = oriented_graph ? 0 : (i+1); j < n; j++) {
            if (i != j) {               
                bool add_edge = (rand() % 100) <= DENSITY;
                int val = (rand() % MAX_COST) + MIN_COST;
                matrix[i][j] = add_edge ? val : INF;
                if (! oriented_graph) {
                    //non-oriented graph
                    matrix[j][i] = matrix[i][j];
                }        
            }
        }
    }
}

// ---------------------------------------------------------------
// ARRAY MATRIX FUNCTIONS VARIANTS

void print_arr_matrix(int *matrix, int m, int n) {
    printf("[\n");
    for (int i = 0; i < m; i++) {
        printf("  ");
        print_array(&(matrix[i*n]), n);
    }
    printf("]\n");
}

bool same_arr_matrixes(int *matrix_1, int *matrix_2, int m, int n, bool oriented_graph) {
    for (int i = 0; i < m; i++) {
        for (int j = oriented_graph ? 0 : (i+1); j < n; j++) {
            if(matrix_1[i*n + j] != matrix_2[i*n + j]) return false;
        }
    }
    return true;
}

int* allocate_arr_matrix(int m, int n) {
    return (int *) malloc(sizeof(int) * m * n);
}

void populate_arr_adj_matrix(int* arr_matrix, int n, int seed, bool oriented_graph) {
    for (int i = 0; i < n; i++) {
        arr_matrix[i*n + i] = 0;
        for (int j = oriented_graph ? 0 : (i+1); j < n; j++) {
            if (i != j) {        
                printf("%d %d", i, j);           
                bool add_edge = (rand() % 100) <= DENSITY;
                int val = (rand() % MAX_COST) + MIN_COST;
                arr_matrix[i*n + j] = add_edge ? val : INF;
                if (! oriented_graph) {
                    //non-oriented graph
                    arr_matrix[j*n + i] = arr_matrix[i*n + j];
                }
            }
        }
    }
}