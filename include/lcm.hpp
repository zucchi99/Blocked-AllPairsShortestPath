#ifndef LCM_HPP
#define LCM_HPP

int lcm(int x, int y) {

    int r=0;

    while (r%x!=0 || r%y!=0) {
        r++;
    }
    
    return r;
}

#endif