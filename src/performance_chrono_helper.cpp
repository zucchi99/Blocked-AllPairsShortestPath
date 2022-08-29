#include "../include/performance_chrono_helper.hpp"

#include <chrono>
#include <vector>
#include <cmath>

using namespace std;
using namespace chrono;

// --------------------------------------------------------------------
// RESOLUTION

// calculates the resolution (takes median value of n executions)
duration<double> resolution() {
    int n = 101;
    vector<duration<double>> res = vector<duration<double>>(n);
    steady_clock::time_point start, end;
    for (int i = 0; i < n; i++) {
        start = steady_clock::now();
        do {
            end = steady_clock::now();
        } while (end == start);
        res[i] = duration_cast<duration<double>>(end - start);
    }
    return res[median(res)];
}

// ----------------------------------------------------------------------
// ARRAY UTILITIES

// swap elements in array
void swap(vector<duration<double>> vec, int start, int finish) {
    duration<double> temp = vec[start];
    vec[start] = vec[finish];
    vec[finish] = temp;
}

// ----------------------------------------------------------------------
// MATH FUNCTIONS

// returns value of mean
double mean(vector<duration<double>> vec) {
    duration<double> sum = (duration<double>) 0;
    for (auto i : vec) {
        sum += i;
    }
    return sum.count() / vec.size();
}

// returns value of mse
double mean_squared_error(vector<duration<double>> vec) {
    double sum = 0;
    double m = mean(vec);
    for (duration<double> i : vec) {
        sum += pow(i.count() - m, 2);
    }
    return sqrt(sum / vec.size());
}

// returns index of the minimum value in the array between positions p and q
int minimum(vector<duration<double>> vec, int p, int q) {
    int minIndex = p;
    for(int i = p + 1; i < q; i++) {
        if(vec[i] < vec[minIndex]) {
            minIndex = i;
        }
    }
    return minIndex;
}

// returns index of the median value in the array
int median(vector<duration<double>> vec) {
    int medianIdx = 0;
    for(int i = 0; i < vec.size() / 2; i++) {
        int minIdx = minimum(vec, i, vec.size() - 1);
        swap(vec, i, minIdx);
        medianIdx++;
    }
    return medianIdx;
}
