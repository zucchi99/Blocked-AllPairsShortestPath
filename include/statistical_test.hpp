#ifndef STATISTICAL_TEST_HPP
#define STATISTICAL_TEST_HPP

#define RANDOM_SEED     0
#define RANDOM_CONSTANT 0.0


struct MultiSizeTestParameters {
    void (*function_to_test) (int* arr_matrix, int n, int b) = NULL;
    int start_input_size        = 4;                // min n to test
    int end_input_size          = 1024;             // max n to test
    double costant_multiplier   = RANDOM_CONSTANT;  // costant used to linearly increase input size (pass RANDOM_CONSTANT to generate a random costant_multiplier)
    int seed                    = RANDOM_SEED;      // seed for input generation (pass RANDOM_SEED to generate a random seed)
    int n_tests_per_round       = 500;              // number of different test foreach given couple (n,B)
    int print_progress_perc     = 1;                // print progress of a test for a given couple (n,B) (for ex. 4 ==> 100/4 = 25% ==> print progress at 25%, 50%, 75%)
    bool stop_current_if_fail   = true;             // true ==> if found an error for a given couple (n,B): stop test this couple but keep testing other couples
    bool stop_all_if_fail       = false;            // true ==> if found an error for a given couple (n,B): stop all tests (return) (NB: stop_all_if_fail ==> stop_current_if_fail but not viceversa)
    bool print_failed_tests     = true;             // true ==> if found an error print seed and the index of the test
};


//default parameters
/*
extern MultiSizeTestParameters default_params = {
    .function_to_test       = NULL,
    .start_input_size       = 4,
    .end_input_size         = 1024,
    .costant_multiplier     = RANDOM_CONSTANT,
    .seed                   = RANDOM_SEED,
    .n_tests_per_round      = 500,
    .print_progress_perc    = 4,
    .stop_current_if_fail   = true,
    .stop_all_if_fail       = false,
    .print_failed_tests     = true
};
*/

int multi_size_statistical_test(MultiSizeTestParameters params);
void print_multi_size_test_parameters(MultiSizeTestParameters params);

// -------------------------------------------------------------------------------------------------
    
int do_arr_floyd_warshall_statistical_test(
    void (*function_to_test) (int* arr_matrix, int n, int b), 
    int input_size, int blocking_factor, 
    int n_tests, int use_always_seed, 
    bool stop_if_fail, int progress_print_fraction, bool print_failed_tests);
    
    
// -------------------------------------------------------------------------------------------------


bool test_arr_floyd_warshall(
    void (*function_to_test) (int* arr_matrix, int n, int b), 
    int *input_instance, int *test_instance_space, 
    int input_size, int blocking_factor);



#endif