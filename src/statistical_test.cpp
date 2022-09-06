
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cassert>
#include <cstring>

#include "../include/adj_matrix_utils.hpp"
#include "../include/generate_n_b_couples.hpp"
#include "../include/host_floyd_warshall.hpp"
#include "../include/statistical_test.hpp"

bool exec_single_single_statistical_test(ExecSingleSizeTestParameters params) {

    //just a rename of the same pointer
    int *f_input = params.input_instance;

    //define an input copy and allocate memory its memory
    int *g_input = allocate_arr_matrix(params.input_size, params.input_size);

    //make a copy of f_input to g_input
    memcpy((void*) g_input, (void*) f_input, params.input_size * params.input_size * sizeof(int));
    
    //classic floyd_warshall on host, used to compare output
    params.f(f_input, params.input_size, params.blocking_factor);

    //function to test execution
    params.g(g_input, params.input_size, params.blocking_factor);

    //return true <==> foreach 0 <= i,j < n : input[i,j] = test[i,j]
    return same_arr_matrixes(f_input, g_input, params.input_size, params.input_size, false);
}


int call_single_size_statistical_test(CallSingleSizeTestParameters params) {


    printf("Test started. Number of tests: %d, matrix size: %d, blocking factor: %d\n", params.n_tests, params.input_size, params.blocking_factor);

    int n_wrong = 0;

    //matrix initialization
    int *input_instance = allocate_arr_matrix(params.input_size, params.input_size);

    int i = 0;
    for (; i < params.n_tests; i++)
    {
        // Progression status print
        if((i > 0) && (i % (params.n_tests / params.print_progress_perc) == 0)) {
            double perc = ((double) i) / ((double) params.n_tests);
            printf("%d%%: %d of %d\n", (int) (perc*100), i, params.n_tests);
        }
        
        // if necessary, generate (pseudo) random input instance
        params.seed = (params.seed == RANDOM_SEED) ? clock() : params.seed;
        
        populate_arr_adj_matrix(input_instance, params.input_size, params.seed, false);

        //define exec single test params
        ExecSingleSizeTestParameters exec_params;
        // most of parameters are copied as received:
        exec_params.f               = params.f;
        exec_params.g               = params.g;
        exec_params.input_size      = params.input_size;
        exec_params.blocking_factor = params.blocking_factor;
        // input instance is allocated and populated here (based on seed value)
        exec_params.input_instance  = input_instance; 

        // perform test
        if (!exec_single_single_statistical_test(exec_params)) {

            n_wrong++;

            if (params.print_failed_tests) printf("%d/%d)\tseed: %d --> ERROR!\n", i, params.n_tests, params.seed);
            
            if (params.stop_current_if_fail) break;
        }
    }

    free(input_instance);

    printf("Test ended. Performed %d/%d tests and got %d/%d errors\n", i, params.n_tests, n_wrong, params.n_tests);

    return n_wrong;
}

int multi_size_statistical_test(MultiSizeTestParameters params) {

    assert(params.end_input_size >= params.start_input_size);
    // stop_all_if_fail ==> stop_current_if_fail
    params.stop_current_if_fail = params.stop_current_if_fail || params.stop_all_if_fail;
    
    int seed = time(NULL);
    srand(seed);
    
    // calculate default values
    double rand_to_multiply = OBTAIN_VAL_TO_MULTIPLY(rand());
    int rand_to_sum = OBTAIN_VAL_TO_SUM(rand());
    // if parameter is random then use calculated values, else use parameter
    params.to_multiply = (params.to_multiply == RANDOM_CONSTANT) ? rand_to_multiply : params.to_multiply;
    params.to_sum =      (params.to_sum      == RANDOM_CONSTANT) ? rand_to_sum      : params.to_sum;
    // in case is used the parameter value, check if it is non-negative
    assert(params.to_multiply >= 0);
    assert(params.to_sum      >= 0);
    
    int n_err_tot = 0;

    std::vector<std::pair<int, int>> list_of_all_n_b;
    list_of_all_n_b = generate_list_of_all_n_b(params.start_input_size, params.end_input_size, 5, params.to_multiply, params.to_sum, params.min_blocking_factor, 50, seed);

    // ------------------------- START PRINTING

    print_multi_size_test_parameters(params);

    printf("List of all (n,B) couples:\n");
    printf("  n\t  B\n");
    for (int i = 0; i < list_of_all_n_b.size(); i++) {  
        printf("%3d\t%3d\n", list_of_all_n_b[i].first, list_of_all_n_b[i].second);
    }
    printf("\n");

    // ------------------------- END PRINTING

    for (int i = 0; i < list_of_all_n_b.size(); i++) {

        int n = list_of_all_n_b[i].first;
        int B = list_of_all_n_b[i].second;

        //printf("n: %d, B: %d\n", n, B);

        //define exec single test params
        CallSingleSizeTestParameters single_test_params;
        // most of parameters are copied as received:
        single_test_params.f                    = params.f;
        single_test_params.g                    = params.g;
        single_test_params.seed                 = params.seed;
        single_test_params.n_tests              = params.n_tests_per_round;
        single_test_params.print_progress_perc  = params.print_progress_perc;
        single_test_params.print_failed_tests   = params.print_failed_tests;
        single_test_params.stop_current_if_fail = params.stop_current_if_fail;
        // the couple (n,B) is calculated here
        single_test_params.input_size      = n;
        single_test_params.blocking_factor = B;

        //execute test
        int n_err = call_single_size_statistical_test(single_test_params);
        
        // count errors
        n_err_tot += n_err;
        if (n_err > 0 && params.stop_all_if_fail) {
            return n_err_tot;
        }
        
        printf("Total number of errors until now: %d (%d new ones)\n\n", n_err_tot, n_err);
        
    }

    return n_err_tot;
}

void print_multi_size_test_parameters(MultiSizeTestParameters params) {
    printf("MultiSizeTestParameters:\n");
    //printf("- pointer to test func:\t%p\n", &params.f);
    //printf("- pointer to comp func:\t%p\n", &params.g);
    printf("- start input size:\t%d\n", params.start_input_size);
    printf("- end input size:\t%d\n", params.end_input_size);

    printf("- costant adder:\t");
    if (params.seed == RANDOM_CONSTANT) printf("RANDOM\n");
    else                                printf("%d\n", params.to_sum);

    printf("- costant multiplier:\t");
    if (params.seed == RANDOM_CONSTANT) printf("RANDOM\n");
    else                                printf("%f\n", params.to_multiply);

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

void print_call_single_size_statistical_test_parameters(CallSingleSizeTestParameters params) {
    printf("CallSingleSizeTestParameters:\n");
    //printf("- pointer to test func:\t%p\n", &params.f);
    //printf("- pointer to comp func:\t%p\n", &params.g);
    printf("- input size:\t%d\n", params.input_size);
    printf("- blocking factor:\t%d\n", params.blocking_factor);
    printf("- seed:\t\t\t");
    if (params.seed == RANDOM_SEED)     printf("RANDOM\n");
    else                                printf("%d\n", params.seed);
    printf("- n tests:\t\t%d\n", params.n_tests);
    printf("- print progress perc:\t%d%%\n", (100 / params.print_progress_perc));
    printf("- stop current if fail:\t%s\n", bool_to_string(params.stop_current_if_fail));
    printf("- print failed tests:\t%s\n", bool_to_string(params.print_failed_tests));
    printf("\n");
}

void print_exec_single_size_statistical_test_parameters(ExecSingleSizeTestParameters params) {
    printf("ExecSingleSizeTestParameters:\n");
    //printf("- pointer to test func:\t%p\n", &params.f);
    //printf("- pointer to comp func:\t%p\n", &params.g);
    printf("- input size:\t%d\n", params.input_size);
    printf("- blocking factor:\t%d\n", params.blocking_factor);
    //printf("- pointer to input data:\t%p\n", &params.input_instance);
    printf("\n");
}

