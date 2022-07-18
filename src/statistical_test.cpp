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
    int n_tests, bool stop_if_fail, int progress_print_fraction) {

    int n_wrong = 0;

    //matrix initialization
    int *input_instance = (int *) malloc(sizeof(int *) * input_size * input_size);
    int *test_instance_space = (int *) malloc(sizeof(int *) * input_size * input_size);

    for (size_t i = 0; i < n_tests; i++)
    {
        // Progression status print
        if((i > 0) && (i % (n_tests/progress_print_fraction) == 0)) {
            double perc = ((double) i) / ((double) n_tests);
            printf("%d%%: %lu of %d\n", (int) (perc*100), i, n_tests);
        }
        
        // generate (pseudo) random input instance
        int rand_seed = clock();
        populate_arr_graph(input_instance, input_size, rand_seed);

        // perform test
        if (!test_arr_floyd_warshall(*function_to_test, input_instance, test_instance_space, input_size, blocking_factor)) {

            n_wrong++;

            printf("%lu/%d)\tseed: %d --> ERROR!\n", i, n_tests, rand_seed);
            
            if (stop_if_fail) return n_wrong;
        }
    }

    free(input_instance);
    free(test_instance_space);

    printf("%d errors detected\n\n", n_wrong);
    return n_wrong;
}