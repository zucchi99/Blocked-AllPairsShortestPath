
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

    /*
    printf("Performing statistical test with:\n");
    printf("\t%d executions\n", n_tests);
    if (use_always_seed==RANDOM_SEED) {
        printf("\tseed=RANDOM\n");
    } else {
        printf("\tseed=%d\n", use_always_seed);
    }

    printf("\tinput_size=%d\n\tblocking_factor=%d\n\n", input_size, blocking_factor);
    */

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

    printf("Test ended. Performed %d/%d tests and got %d/%d errors\n", i, n_tests, n_wrong, n_tests);
    return n_wrong;
}

int multi_size_statistical_test(MultiSizeTestParameters params) {

    assert(params.end_input_size >= params.start_input_size);
    // stop_all_if_fail ==> stop_current_if_fail
    params.stop_current_if_fail = params.stop_current_if_fail || params.stop_all_if_fail;
    
    // outputs a random number between 1.300 and 1.600
    srand(time(NULL));
    double rand_costant_multiplier = ((double) ((rand() % 300) + 1300)) / ((double) 1000);
    params.costant_multiplier = (params.costant_multiplier == RANDOM_CONSTANT) ? rand_costant_multiplier : params.costant_multiplier;

    printf("Performing Multi-size statistical test:\n");
    printf("- Input sizes between %d and %d, increase is linear, using %f as costant multiplier\n", params.start_input_size, params.end_input_size, params.costant_multiplier);
    printf("- Blocking factor are generated randomly between [1, n/2] U {n}\n");
    printf("- Number of executions for each couple (n,B) used: %d\n\n", params.n_tests_per_round);

    int n_err_tot = 0;

    for (int n = params.start_input_size; n <= params.end_input_size; n = max(((int) (params.costant_multiplier * (double) n)), (n+1)) ) { // n *= 2

        // use max 5 different blocking factors
        int B_used[5];
        // initially all are -1
        for (int i = 0; i < 5; i++) B_used[i] = -1;

        // index of the currently used B 
        int cur_B_idx = -1;

        // generate randomly B check if it is a divisor of n and not already used.
        // generate  maximum 50 random B (necessary to avoid non-termination, maybe n is prime)
        for (int tests = 0; tests < 50 && cur_B_idx < 5; tests++) {

            // range for b is between 0 and n/2
            int B = rand() % min(n/2, MAX_BLOCKING_FACTOR);
            // but if it is zero then use B=n
            B = (B == 0) ? n : B; 

            // test if it is ok to be executed (b is a new divisor)
            bool exec_cond = (n % B == 0) && (B <= MAX_BLOCKING_FACTOR);
            for (int i = 0; (i <= cur_B_idx) && exec_cond; i++) exec_cond = (B != B_used[i]);

            // if((n % BLOCKING_FACTOR) == 0) {
            if (exec_cond) {

                // add b to the list of B used
                B_used[++cur_B_idx] = B;
                //print n and B
                printf("n: %d, B: %d\n", n, B);

                //execute test
                int n_err = do_arr_floyd_warshall_statistical_test(
                    params.function_to_test, n, B, params.n_tests_per_round, params.seed, params.stop_current_if_fail, params.print_progress_perc, params.print_failed_tests);
                // int n_err = do_arr_floyd_warshall_statistical_test(&arr_floyd_warshall_blocked, n, BLOCKING_FACTOR, 1000, RANDOM_SEED, true, 4, true);
                
                // count errors
                n_err_tot += n_err;
                if (n_err > 0 && params.stop_all_if_fail) {
                    return n_err_tot;
                }
                
                printf("Cumulative errors at size=%d, blocking_factor=%d: %d (%d new ones)\n\n", n, B, n_err_tot, n_err);
            }

        }
    }

    return n_err_tot;
}

void print_multi_size_test_parameters(MultiSizeTestParameters params) {
    printf("MultiSizeTestParameters:\n");
    printf("- pointer to test func:\t%p\n", &params.function_to_test);
    printf("- start input size:\t%d\n", params.start_input_size);
    printf("- end input size:\t%d\n", params.end_input_size);
    printf("- costant multiplier:\t");
    if (params.seed == RANDOM_CONSTANT) printf("RANDOM\n");
    else                                printf("%f\n", params.costant_multiplier);
    printf("- seed:\t\t\t");
    if (params.seed == RANDOM_SEED)     printf("RANDOM\n");
    else                                printf("%d\n", params.seed);
    printf("- n tests per round:\t%d\n", params.n_tests_per_round);
    printf("- print progress perc:\t%d%%\n", (100 / params.print_progress_perc));
    printf("- stop current if fail:\t%s\n", bool_to_string(params.stop_current_if_fail));
    printf("- stop all if fail:\t%s\n", bool_to_string(params.stop_all_if_fail));
    printf("- print failed tests:\t%s\n", bool_to_string(params.print_failed_tests));
    printf("- blocking factor:\tRANDOM\n");
    printf("\n");
}
