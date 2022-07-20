#ifndef DEVICE_FLOYD_WARSHALL_V1_2_CUH
#define DEVICE_FLOYD_WARSHALL_V1_2_CUH

#define MAX_BLOCK_SIZE 1024 // in realtà basta fare le proprerties della macchina

/// Macro to get block starting position (of a column or of a row)
#define BLOCK_START(block_index,B) (block_index * B)

/// Macro to get block ending position (of a column or of a row)
#define BLOCK_END(block_index,B) ((block_index+1) * B)

__global__ void execute_round_device_v1_2_phase_1(int *matrix, int n, int t, int B);
__global__ void execute_round_device_v1_2_phase_2(int *matrix, int n, int t, int B);
__global__ void execute_round_device_v1_2_phase_3(int *matrix, int n, int t, int B);

void floyd_warshall_blocked_device_v1_2(int *matrix, int n, int B);

#endif