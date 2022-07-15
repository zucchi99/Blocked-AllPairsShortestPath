#ifndef ADJ_MATRIX_UTILS_HPP
#define ADJ_MATRIX_UTILS_HPP

#include <stdbool.h>

/// Big M, value that should be threated as "infinity"
#define INF __INT16_MAX__

/// Parameters used when generating a graph
#define DENSITY 60
#define MIN_COST 1
#define MAX_COST 20

/// Print a bool as a string
#define bool_to_string(cond) (cond ? "true" : "false")

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
int* generate_arr_graph(int n, int seed);
bool same_arr_matrix(int *matrix_1, int *matrix_2, int n);

#endif // ADJ_MATRIX_UTILS_H