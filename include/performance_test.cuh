#ifndef PERFORMANCE_TEST_CUH
#define PERFORMANCE_TEST_CUH

#include <chrono>
#include <string>

void do_nvprof_performance_test(void (*floyd_warshall_arr_algorithm)(int * matrix, int n, int B), int input_size, int blocking_factor, int number_of_tests, int seed);

void do_chrono_performance_test(void (*floyd_warshall_arr_algorithm)(int * matrix, int n, int B), int input_size, int blocking_factor, int number_of_tests, int seed, std::string version, std::string output_file);
std::chrono::duration<double> initialization_time(int input_size, int repetitions, int seed);

#endif