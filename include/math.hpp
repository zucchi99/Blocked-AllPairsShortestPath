#ifndef MATH_HPP
#define MATH_HPP

#include <chrono>
#include <vector>

// machine resolution
//unsigned long resolution_nanoseconds();

void swap(std::vector<double> &vec, int i, int j);

// values of
double mean(std::vector<double> &vec);
double mean_squared_error(std::vector<double> &vec);

// indexes of
int minimum(std::vector<double> &vec, int p, int q);
int median(std::vector<double> &vec);


#endif
