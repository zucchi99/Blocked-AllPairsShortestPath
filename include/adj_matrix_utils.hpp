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

// ---------------------------------------------------------------
// MATRIX GENERATION, COMPARE and others utils

int** generate_graph(int n, int seed);
bool same_matrix(int **matrix_1, int **matrix_2, int m, int n);

// ---------------------------------------------------------------
// ARRAY MATRIX FUNCTIONS VARIANTS

void print_arr_matrix(int *matrix, int m, int n);
void populate_arr_graph(int* arr_matrix, int n, int seed);
void copy_arr_graph(int* src, int* target, int n);
bool same_arr_matrix(int *matrix_1, int *matrix_2, int n);

#endif // ADJ_MATRIX_UTILS_H