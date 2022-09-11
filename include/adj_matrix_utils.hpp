#ifndef ADJ_MATRIX_UTILS_HPP
#define ADJ_MATRIX_UTILS_HPP

#include <stdbool.h>

#include "macros.hpp"

/// Parameters used when generating a graph
#define DENSITY 60
#define MIN_COST 1
#define MAX_COST 20

// ---------------------------------------------------------------
//  PRINT UTILS

void print_array(int *array, int size);
void print_matrix(int **matrix, int m, int n);
void print_element(int val, int infinity);
void print_arr_matrix(int *matrix, int m, int n);

// ---------------------------------------------------------------
// MATRIX GENERATION, COMPARE and others utils

int** allocate_matrix(int m, int n);
void populate_adj_matrix(int **matrix, int n, int seed, bool oriented_graph);
bool same_matrixes(int **matrix_1, int **matrix_2, int m, int n, bool oriented_graph);
void copy_arr_matrix(int *dest_matrix, int *source_matrix, int m, int n);

// ---------------------------------------------------------------
// ARRAY MATRIX FUNCTIONS VARIANTS

int* allocate_arr_matrix(int m, int n);
void populate_arr_adj_matrix(int* arr_matrix, int n, int seed, bool oriented_graph);
bool same_arr_matrixes(int *matrix_1, int *matrix_2, int m, int n, bool oriented_graph);

#endif // ADJ_MATRIX_UTILS_H