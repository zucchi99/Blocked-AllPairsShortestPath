#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../include/cuda_errors_utils.cuh"

void handle_error(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
        exit(EXIT_FAILURE);
    }
}

void check_CUDA_error(const char *msg) {
    cudaError_t err = cudaGetLastError();
    if(cudaSuccess != err) {
        fprintf(stderr, "ERRORE CUDA: >%s<: >%s<. Eseguo: EXIT\n", msg, cudaGetErrorString(err) );
        exit(-1);
    }
}