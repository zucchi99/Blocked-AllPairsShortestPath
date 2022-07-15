#ifndef NUM_MACRO_HPP
#define NUM_MACRO_HPP

/// Big M, value that should be threated as "infinity"
#define INF __INT16_MAX__

/// Get minimum of two values
#define min(a,b) ((a < b) ? a : b)

/// Sum two numbers if they are not infinite, else return infinity
#define sum_if_not_infinite(a,b,infinity) ((a==infinity) || (b==infinity)) ? infinity : a+b 


#endif