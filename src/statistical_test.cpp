#include "../include/statistical_test.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "../include/host_floyd_warshall.hpp"
#include "../include/adj_matrix_utils.hpp"


bool test_arr_floyd_warshall(
    void (*function_to_test) (int* arr_matrix, int n, int b), 
    int *input_instance, int *test_instance_space, 
    int input_size, int blocking_factor) {

    copy_arr_graph(input_instance, test_instance_space, input_size);
    
    // correct floyd_warshall execution
    arr_floyd_warshall(test_instance_space, input_size);

    // function to test execution
    function_to_test(input_instance, input_size, blocking_factor);

    return same_arr_matrix(input_instance, test_instance_space, input_size);
}


int do_arr_floyd_warshall_statistical_test(
    void (*function_to_test) (int* arr_matrix, int n, int b), 
    int input_size, int blocking_factor, 
    int n_tests, int use_always_seed, 
    bool stop_if_fail, int progress_print_fraction, bool print_failed_tests) {

    printf("Performing statistical test with:\n");
    printf("\t%d executions\n", n_tests);
    if (use_always_seed==RANDOM_SEED) {
        printf("\tseed=RANDOM\n");
    } else {
        printf("\tseed=%d\n", use_always_seed);
    }

    printf("\tinput_size=%d\n\tblocking_factor=%d\n\n", input_size, blocking_factor);
    
    int n_wrong = 0;

    //matrix initialization
    int *input_instance = (int *) malloc(sizeof(int *) * input_size * input_size);
    int *test_instance_space = (int *) malloc(sizeof(int *) * input_size * input_size);

    int i;
    for (i = 0; i < n_tests; i++)
    {
        // Progression status print
        if((i > 0) && (i % (n_tests/progress_print_fraction) == 0)) {
            double perc = ((double) i) / ((double) n_tests);
            printf("%d%%: %d of %d\n", (int) (perc*100), i, n_tests);
        }
        
        // if necessary, generate (pseudo) random input instance
        int seed = (use_always_seed == RANDOM_SEED) ? clock() : use_always_seed;
        
        populate_arr_graph(input_instance, input_size, seed);

        // perform test
        if (!test_arr_floyd_warshall(*function_to_test, input_instance, test_instance_space, input_size, blocking_factor)) {

            n_wrong++;

            if (print_failed_tests) printf("%d/%d)\tseed: %d --> ERROR!\n", i, n_tests, seed);
            
            if (stop_if_fail) break;
        }
    }

    free(input_instance);
    free(test_instance_space);

    printf("Test ended. Performed %d/%d tests and got %d/%d errors\n\n", i, n_tests, n_wrong, n_tests);
    return n_wrong;
}