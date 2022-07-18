#ifndef CUDA_ERRORS_UTILS_CUH
#define CUDA_ERRORS_UTILS_CUH


#define HANDLE_ERROR(err) (handle_error(err, __FILE__, __LINE__))

void handle_error(cudaError_t err, const char *file, int line);

void check_CUDA_error(const char *msg);

#endif