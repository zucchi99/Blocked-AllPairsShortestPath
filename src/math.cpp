#include "../include/math.hpp"

//#include <chrono>
#include <vector>
#include <cmath>

using namespace std;
//using namespace chrono;

// --------------------------------------------------------------------
// RESOLUTION

// calculates the resolution (takes median value of n executions)
/*
unsigned long resolution_nanoseconds() {
    int n = 101;
    auto res = std::vector<duration<int64_t, nano>>(n);
    steady_clock::time_point start, end;
    for (int i = 0; i < n; i++) {
        start = steady_clock::now();
        do {
            end = steady_clock::now();
        } while (end == start);
        res[i] = (end - start);
    }
    return res[median(res)].count();
}
*/
    
// ----------------------------------------------------------------------
// ARRAY UTILITIES

// swap elements in array
void swap(std::vector<double> &vec, int i, int j) {
    double temp = vec[i];
    vec[i] = vec[j];
    vec[j] = temp;
}

// ----------------------------------------------------------------------
// MATH FUNCTIONS

// returns value of mean
double mean(std::vector<double> &vec) {
    double sum = 0;
    for (auto i : vec) {
        sum += i;
    }
    return sum / vec.size();
}

// returns value of mse
double mean_squared_error(std::vector<double> &vec) {
    double sum = 0;
    double m = mean(vec);
    for (auto i : vec) {
        sum += pow(i - m, 2);
    }
    return sqrt(sum / vec.size());
}

// returns index of the minimum value in the array between positions p and q
int minimum(std::vector<double> &vec, int p, int q) {
    int minIndex = p;
    for(int i = p + 1; i < q; i++) {
        if(vec[i] < vec[minIndex]) {
            minIndex = i;
        }
    }
    return minIndex;
}

// returns index of the median value in the array
int median(std::vector<double> &vec) {
    int medianIdx = 0;
    for(int i = 0; i < vec.size() / 2; i++) {
        int minIdx = minimum(vec, i, vec.size() - 1);
        swap(vec, i, minIdx);
        medianIdx++;
    }
    return medianIdx;
}


