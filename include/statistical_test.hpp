#ifndef STATISTICAL_TEST_HPP
#define STATISTICAL_TEST_HPP

#define RANDOM_SEED     0
#define RANDOM_CONSTANT 0.0


struct MultiSizeTestParameters {
    void (*f) (int* arr_matrix, int n, int b) = NULL;   // test function
    void (*g) (int* arr_matrix, int n, int b) = NULL;   // compare function
    int start_input_size        = 4;                    // min n to test
    int end_input_size          = 1024;                 // max n to test
    double costant_multiplier   = RANDOM_CONSTANT;      // costant used to linearly increase input size (pass RANDOM_CONSTANT to generate a random costant_multiplier)
    int seed                    = RANDOM_SEED;          // seed for input generation (pass RANDOM_SEED to generate a random seed)
    int n_tests_per_round       = 500;                  // number of different test foreach given couple (n,B)
    int print_progress_perc     = 1;                    // print progress of a test for a given couple (n,B) (for ex. 4 ==> 100/4 = 25% ==> print progress at 25%, 50%, 75%), if 1 is disabled
    bool stop_current_if_fail   = true;                 // true ==> if found an error for a given couple (n,B): stop test this couple but keep testing other couples
    bool stop_all_if_fail       = false;                // true ==> if found an error for a given couple (n,B): stop all tests (return control) (NB: stop_all_if_fail ==> stop_current_if_fail but not viceversa)
    bool print_failed_tests     = true;                 // true ==> if found an error print seed and the index of the test
    int min_blocking_factor     = 2;                    // the minimum blocking factor you are intrested testing
};

// generates many couples (n,B) and foreach couple generates inputs and executes executes f,g n_tests_per_round times
// returns: total number of errors of all couples
int  multi_size_statistical_test(MultiSizeTestParameters params);
void print_multi_size_test_parameters(MultiSizeTestParameters params);

// -------------------------------------------------------------------------------------------------

struct CallSingleSizeTestParameters {
    void (*f) (int* arr_matrix, int n, int b) = NULL;   // test function
    void (*g) (int* arr_matrix, int n, int b) = NULL;   // compare function
    int input_size              = 256;                  // input test size
    int blocking_factor         = 12;                   // blocking factor test size
    int seed                    = RANDOM_SEED;          // seed for input generation (pass RANDOM_SEED to generate a random seed)
    int n_tests                 = 500;                  // number of different tests to do
    int print_progress_perc     = 1;                    // print progress of a test for a given couple (n,B) (for ex. 4 ==> 100/4 = 25% ==> print progress at 25%, 50%, 75%), if 1 is disabled
    bool stop_current_if_fail   = true;                 // true ==> if found an error: stop testing (return control)
    bool print_failed_tests     = true;                 // true ==> if found an error print seed and the index of the test
};

// given a couple (n,B), generates inputs and executes f,g n_tests_per_round times
// returns: total number of errors of the given couple
int  call_single_size_statistical_test(CallSingleSizeTestParameters params);
void print_call_single_size_statistical_test_parameters(CallSingleSizeTestParameters params);
    
// -------------------------------------------------------------------------------------------------

struct ExecSingleSizeTestParameters {
    void (*f) (int* arr_matrix, int n, int b) = NULL;   // test function
    void (*g) (int* arr_matrix, int n, int b) = NULL;   // compare function
    int input_size              = 256;                  // input test size
    int blocking_factor         = 12;                   // blocking factor test size
    int *input_instance         = NULL;                 // input instance populated
};

// given a couple (n,B) and an an input instance populated executes f,g 1 time
// returns: f(input) == g(input)
bool exec_single_single_statistical_test(ExecSingleSizeTestParameters params);
void print_exec_single_size_statistical_test_parameters(ExecSingleSizeTestParameters params);

#endif