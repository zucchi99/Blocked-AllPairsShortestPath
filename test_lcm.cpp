#include <stdio.h>

#include "include/lcm.hpp"

#define size 8

int main() {
    int x[size] = {1,  1,  2,  3,  4,  5,  6,  10};
    int y[size] = {1,  2,  3,  2,  6,  6,  8,  13};
    int z[size] = {1,  2,  6,  6,  12, 30, 24, 130};

    for (int i=0; i<size; i++) {
        printf("%d)input:%d,%d;\tresult:%d;\texpected:%d\t", i , x[i], y[i], lcm(x[i], y[i]), z[i]);
        if (lcm(x[i], y[i]) == z[i]) {
            printf("ok\n");
        } else {
            printf("ERROR\n");
        }
    }
}

