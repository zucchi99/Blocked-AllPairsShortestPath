#ifndef MACROS_HPP
#define MACROS_HPP

/// Big M, value that should be threated as "infinity"
#define INF __INT16_MAX__

/// Get minimum of two values
#define min(a,b) ((a < b) ? a : b)

/// Sum two numbers if they are not infinite, else return infinity
#define sum_if_not_infinite(x1,x2,infinity) ((x1==infinity) || (x2==infinity)) ? infinity : (x1+x2)


#define MAX_BLOCK_SIZE 1024 // in realtà basta fare le proprerties della macchina

/// Macro to get block starting position (of a column or of a row)
#define BLOCK_START(block_index,B) (block_index * B)

/// Macro to get block ending position (of a column or of a row)
#define BLOCK_END(block_index,B) ((block_index+1) * B)

/// Print a bool as a string
#define bool_to_string(cond) (cond ? "true" : "false")


#define pitched_pointer(matrix, i, j, pitch) ((int *)((char*) matrix + i * pitch) + j)


#endif