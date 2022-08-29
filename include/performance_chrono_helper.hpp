#ifndef PERFORMANCE_CHRONO_HELPER_CUH
#define PERFORMANCE_CHRONO_HELPER_CUH

#include <chrono>
#include <vector>

// machine resolution
std::chrono::duration<double> resolution();

// values of
double mean (std::vector<std::chrono::duration<double>> vec);
double mean_squared_error (std::vector<std::chrono::duration<double>> vec);

// indexes of
int minimum(std::vector<std::chrono::duration<double>> vec, int p, int q);
int median(std::vector<std::chrono::duration<double>> vec);

#endif