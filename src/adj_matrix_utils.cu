
#include <stdio.h>
#include <stdlib.h>

#include "../include/adj_matrix_utils.cuh"

__device__ void print_matrix_device(int *matrix, int m, int n) {
    printf("[\n");
    for (int i = 0; i < m; i++) {
        printf("  ");
        for (int j = 0; j < n; j++) {
            int val = matrix[i*n + j];
            if (val < INF)
                printf("%02d", val);
            else 
                printf("--");
            if (j < n-1) printf(", ");
        }
        printf("\n");
    }
    printf("]\n");
}
