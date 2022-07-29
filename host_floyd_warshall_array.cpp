
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

    int BLOCKING_FACTOR = 2;

    //memory allocation 
    // int *rand_matrix_1 = (int *) malloc(sizeof(int *) * n * n);
    // int *rand_matrix_2 = (int *) malloc(sizeof(int *) * n * n);

    //random seed
    int rand_seed = time(NULL);
    printf("seed: %d\n", rand_seed);
    srand(rand_seed);

    //matrix initialization
    int *rand_matrix_1 = generate_arr_graph(n, rand_seed);

    //matrix print
    printf("input adjacency matrix %lux%lu:\n", n, n);
    print_arr_matrix(rand_matrix_1, n, n);

    //floyd_warshall execution
    arr_floyd_warshall(rand_matrix_1, n);

    //print floyd_warshall output
    printf("output adjacency matrix classic %lux%lu:\n", n, n);
    print_arr_matrix(rand_matrix_1, n, n);

    //---------------------------------------------------------------

    //matrix initialization with same seed
    int *rand_matrix_2 = generate_arr_graph(n, rand_seed);
    
    //floyd_warshall_blocked execution
    arr_floyd_warshall_blocked(rand_matrix_2, n, BLOCKING_FACTOR);
    
    //print floyd_warshall_blocked output
    printf("output adjacency matrix blocked %lux%lu:\n", n, n);
    print_arr_matrix(rand_matrix_2, n, n);

    //---------------------------------------------------------------

    //compare matrixes output
    bool are_the_same = same_arr_matrix(rand_matrix_1, rand_matrix_2, n);
    printf("Matrixes are equal? %s\n", bool_to_string(are_the_same));

    return 0;
}
