#ifndef PERFORMANCE_TEST_CUH
#define PERFORMANCE_TEST_CUH

void do_nvprof_performance_test(void (*floyd_warshall_arr_algorithm)(int * matrix, int n, int B), int input_size, int blocking_factor, int number_of_tests, int seed);


#endif