#ifndef NUM_MACRO_HPP
#define NUM_MACRO_HPP

/// Big M, value that should be threated as "infinity"
#define INF __INT16_MAX__

/// Get minimum of two values
#define min(a,b) ((a < b) ? a : b)

/// Sum two numbers if they are not infinite, else return infinity
#define sum_if_not_infinite(x1,x2,infinity) ((x1==infinity) || (x2==infinity)) ? infinity : x1+x2 


#endif