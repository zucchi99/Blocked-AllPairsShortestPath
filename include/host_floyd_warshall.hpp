#ifndef HOST_FLOYD_WARSHALL
#define HOST_FLOYD_WARSHALL

// #define min(a,b) ((a < b) ? a : b)

#include "num_macro.hpp"

int sum_if_not_infinite(int a, int b, int infinity);

// ---------------------------------------------------------------------------
// Matrix data structure version

void floyd_warshall(int **matrix, int n);
void floyd_warshall_blocked(int **matrix, int n, int B);
void execute_round(int **matrix, int n, int t, int row, int col, int B);

// ---------------------------------------------------------------------------
// Array data structure version


void arr_floyd_warshall(int *matrix, int n);
void arr_floyd_warshall_blocked(int *matrix, int n, int B);
// int sum_if_not_infinite(int a, int b, int infinity);
void arr_execute_round(int *matrix, int n, int t, int row, int col, int B);

#endif