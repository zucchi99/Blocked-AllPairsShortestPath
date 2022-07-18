#ifndef STATISTICAL_TEST_HPP
#define STATISTICAL_TEST_HPP

# define RANDOM_SEED 0


bool test_arr_floyd_warshall(
    void (*function_to_test) (int* arr_matrix, int n, int b), 
    int *input_instance, int *test_instance_space, 
    int input_size, int blocking_factor);


int do_arr_floyd_warshall_statistical_test(
    void (*function_to_test) (int* arr_matrix, int n, int b), 
    int input_size, int blocking_factor, 
    int n_tests, int use_always_seed, 
    bool stop_if_fail, int progress_print_fraction, bool print_failed_tests);


int multi_size_statistical_test(
    void (*function_to_test)  (int* arr_matrix, int n, int b), 
    int start_input_size, int end_input_size, 
    int min_blocking_factor, int max_blocking_factor, 
    int n_tests_per_round, int use_always_seed, 
    bool stop_if_fail, bool print_failed_tests);

#endif