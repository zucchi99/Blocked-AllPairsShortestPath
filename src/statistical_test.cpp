
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cassert>
#include <cstring>

#include "../include/adj_matrix_utils.hpp"
#include "../include/host_floyd_warshall.hpp"
#include "../include/statistical_test.hpp"

bool test_arr_floyd_warshall(
    void (*function_to_test) (int* arr_matrix, int n, int b), 
    int *input_instance, int *test_instance_space, 
    int input_size, int blocking_factor) {

    //make a copy of input_instance to test_instance
    memcpy((void*) test_instance_space, (void*) input_instance, input_size*input_size*sizeof(int));
    
    //classic floyd_warshall on host, used to compare output
    host_array_floyd_warshall(test_instance_space, input_size);

    //function to test execution
    function_to_test(input_instance, input_size, blocking_factor);

    //return true <==> foreach 0 <= i,j < n : input[i,j] = test[i,j]
    return same_arr_matrixes(input_instance, test_instance_space, input_size, input_size, false);
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
        
        populate_arr_adj_matrix(input_instance, input_size, seed, false);

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

int multi_size_statistical_test(
    void (*function_to_test)  (int* arr_matrix, int n, int b), 
    int start_input_size, int end_input_size, 
    int min_blocking_factor, int max_blocking_factor, 
    int n_tests_per_round, int use_always_seed, 
    bool stop_if_fail, bool print_failed_tests) {

    // assert(end_input_size%start_input_size==0);
    assert(end_input_size>=start_input_size);
    // assert(start_input_size%2==0);
    // assert(end_input_size%2==0);

    // assert(max_blocking_factor%min_blocking_factor==0);
    assert(max_blocking_factor>=min_blocking_factor);
    // assert(min_blocking_factor%2==0);
    // assert(max_blocking_factor%2==0);

    // assert(start_input_size%min_blocking_factor==0);

    printf("Performing Multi-size statistical test:\n");
    printf("\tFrom %d to %d input size (multiplying *2 every time)", start_input_size, end_input_size);
    printf("\tApplying from %d to %d blocking factor for each size (multiplying *2 every time)", min_blocking_factor, max_blocking_factor);
    printf("\t%d Executions for each single test round\n", n_tests_per_round);

    if (use_always_seed==RANDOM_SEED) {
        printf("\tseed=RANDOM\n");
    } else {
        printf("\tseed=%d\n", use_always_seed);
    }

    int n_err_tot = 0;

    // outputs a random number between 1.100 and 1.600
    srand(time(NULL));
    double linear_increase = ((double) ((rand() % 500) + 1100)) / ((double) 1000);
    printf("constant of linear increase for n: %f\n", linear_increase);

    for (int n = start_input_size; n <= end_input_size; n = (int) (linear_increase * (double) n)) { // n *= 2

        int MAX_B = min(n, max_blocking_factor);
    
        //for (int BLOCKING_FACTOR = min_blocking_factor; BLOCKING_FACTOR <= MAX_B; BLOCKING_FACTOR *= 2) {
        
        // use max 5 different blocking factors
        int B[5];
        // initially all are -1
        for (int i = 0; i < 5; i++) B[i] = -1;

        // index of the currently used B 
        int cur_B_idx = -1;

        // generate randomly B check if it is a divisor of n and not already used.
        // generate  maximum 50 random B (necessary to avoid non-termination, maybe n is prime)
        for (int tests = 0; tests < 50 && cur_B_idx < 5; tests++) {

            // range for b is between 0 and n/2
            int b = rand() % (n/2);
            // but if it is zero then use B=n
            b = (b == 0) ? n : b;       

            // test if it is ok to be executed (b is a new divisor)
            bool exec_cond = (n%b == 0) && (b <= MAX_BLOCKING_FACTOR);
            for (int i = 0; (i <= cur_B_idx) && exec_cond; i++) exec_cond = (b != B[i]);

            int BLOCKING_FACTOR = b;

            // if((n % BLOCKING_FACTOR) == 0) {
            if (exec_cond) {

                // add b to the list of B used
                B[++cur_B_idx] = b;
                //print n and B
                printf("n: %d, B: %d\n", n, BLOCKING_FACTOR);

                //execute test
                int n_err = do_arr_floyd_warshall_statistical_test(
                    function_to_test, n, BLOCKING_FACTOR, n_tests_per_round, use_always_seed, stop_if_fail, 1, print_failed_tests);
                // int n_err = do_arr_floyd_warshall_statistical_test(&arr_floyd_warshall_blocked, n, BLOCKING_FACTOR, 1000, RANDOM_SEED, true, 4, true);
                
                // count errors
                n_err_tot += n_err;
                if (n_err>0 && stop_if_fail) {
                    return n_err_tot;
                    //break;
                };
                
                printf("Cumulative errors at size=%d, blocking_factor=%d:\t%d (%d new ones)\n\n", n, BLOCKING_FACTOR, n_err_tot, n_err);
            }
            // }

        }
    }

    return n_err_tot;
}