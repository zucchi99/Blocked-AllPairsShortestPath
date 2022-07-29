#ifndef HOST_FLOYD_WARSHALL
#define HOST_FLOYD_WARSHALL


#include "macros.hpp"

// ---------------------------------------------------------------------------
// Matrix data structure version

void floyd_warshall(int **matrix, int n);
void floyd_warshall_blocked(int **matrix, int n, int B);
void execute_round(int **matrix, int n, int t, int row, int col, int B);

// ---------------------------------------------------------------------------
// Array data structure version

void arr_floyd_warshall(int *matrix, int n);
void arr_floyd_warshall_blocked(int *matrix, int n, int B);
void arr_execute_round(int *matrix, int n, int t, int row, int col, int B);

#endif