#include "../include/include_needed_libraries.cuh"

int main(int argc, char *argv[]) {

    // the code running is ONLY host
    // BUT the files handle_arguments_and_execute.cu and performance_test.cu in other cases launch can cuda kernels (it depends on which function passed, not this case since it is host)
    // so they must be .cu
    // since g++ don't recognise .cu files, this one is .cu too
    // just compile with nvcc that will call g++ and compile the code

    return handle_arguments_and_execute(argc, argv, (void(*) (int*, int, int)) &host_array_floyd_warshall_blocked);

}
