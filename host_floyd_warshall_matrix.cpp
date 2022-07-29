
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>

#include "include/adj_matrix_utils.hpp"
#include "include/host_floyd_warshall.hpp"

int main() {

    //matrix size n*n
    size_t n = 6;

    //if no weights in graph:
    //int INF = (n * (n-1) / 2) + 1;

    int BLOCKING_FACTOR = 2;

    //memory allocation 
    // int **rand_matrix_1 = (int **) malloc(sizeof(int *) * n);
    // for (int i = 0; i < n; i++) {
    //     rand_matrix_1[i] = (int *) malloc(sizeof(int) * n);
    // }

    // int **rand_matrix_2 = (int **) malloc(sizeof(int *) * n);
    // for (int i = 0; i < n; i++) {
    //     rand_matrix_2[i] = (int *) malloc(sizeof(int) * n);
    // }

    int **rand_matrix_1;
    int **rand_matrix_2;

    //random seed
    int rand_seed = time(NULL);
    printf("seed: %d\n", rand_seed);
    srand(rand_seed);

    //matrix initialization
    rand_matrix_1 = generate_graph(n, rand_seed);

    //matrix print
    printf("input adjacency matrix %lux%lu:\n", n, n);
    print_matrix(rand_matrix_1, n, n);

    //floyd_warshall execution
    floyd_warshall(rand_matrix_1, n);

    //print floyd_warshall output
    printf("output adjacency matrix classic %lux%lu:\n", n, n);
    print_matrix(rand_matrix_1, n, n);

    //---------------------------------------------------------------

    //matrix initialization with same seed
    rand_matrix_2 = generate_graph(n, rand_seed);
    
    //floyd_warshall_blocked execution
    floyd_warshall_blocked(rand_matrix_2, n, BLOCKING_FACTOR);
    
    //print floyd_warshall_blocked output
    printf("output adjacency matrix blocked %lux%lu:\n", n, n);
    print_matrix(rand_matrix_2, n, n);

    //---------------------------------------------------------------

    //compare matrixes output
    bool are_the_same = same_matrix(rand_matrix_1, rand_matrix_2, n, n);
    printf("Matrixes are equal? %s\n", bool_to_string(are_the_same));

    return 0;
}
