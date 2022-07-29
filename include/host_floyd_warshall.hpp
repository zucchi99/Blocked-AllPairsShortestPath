#ifndef HOST_FLOYD_WARSHALL
#define HOST_FLOYD_WARSHALL


#include "macros.hpp"

// ---------------------------------------------------------------------------
// Matrix data structure version

void host_matrix_floyd_warshall(int **matrix, int n);
void host_matrix_floyd_warshall_blocked(int **matrix, int n, int B);
void host_matrix_execute_round(int **matrix, int n, int t, int row, int col, int B);

// ---------------------------------------------------------------------------
// Array data structure version

void host_array_floyd_warshall(int *matrix, int n);
void host_array_floyd_warshall_blocked(int *matrix, int n, int B);
void host_array_execute_round(int *matrix, int n, int t, int row, int col, int B);

#endif