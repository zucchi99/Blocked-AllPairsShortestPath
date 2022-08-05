#include "../include/lcm.hpp"

int lcm(int x, int y) {

    int lcm = 1;
    while (lcm%x != 0 || lcm%y != 0)
    {
        lcm++;
    }

    return lcm;
}