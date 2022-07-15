#ifndef HOST_FLOYD_WARSHALL
#define HOST_FLOYD_WARSHALL

#define min(a,b) ((a < b) ? a : b)

void floyd_warshall(int **matrix, int n);
void floyd_warshall_blocked(int **matrix, int n, int B);
void execute_round(int **matrix, int n, int t, int row, int col, int B);

int sum_if_not_infinite(int a, int b, int infinity);




#endif