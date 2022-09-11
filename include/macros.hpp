#ifndef MACROS_HPP
#define MACROS_HPP

/// Big M, value that should be threated as "infinity"
#define INF (__INT16_MAX__/2-1)

/// Get minimum or maximum of two values
// renamed from min and max to avoid overload with std::min and std::max
#define mmin(a,b) ((a < b) ? a : b)
#define mmax(a,b) ((a > b) ? a : b)

/// Sum two numbers if they are not infinite, else return infinity
#define sum_if_not_infinite(x1,x2,infinity) ((x1>=infinity) || (x2>=infinity)) ? infinity : (x1+x2)


#define MAX_BLOCK_SIZE 1024 // in realtà basta fare le proprerties della macchina
#define MAX_BLOCKING_FACTOR 32 // 32*32 = 1024

/// Macro to get block starting position (of a column or of a row)
#define BLOCK_START(block_index,B) (block_index * B)

/// Macro to get block ending position (of a column or of a row)
#define BLOCK_END(block_index,B) ((block_index+1) * B)

/// Print a bool as a string
#define bool_to_string(cond) (cond ? "true" : "false")

/// returns the pointer of given (i,j) using the access pattern of a 2D pitched memory
#define pitched_pointer(matrix, i, j, pitch) ( (int *) (((char*) matrix + i * pitch) + j) )


#endif