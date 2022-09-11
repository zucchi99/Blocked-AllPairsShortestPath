# Blocked-AllPairsShortestPath

In this repository we propose our implementation for the Blocked Floyd Warshall APSP algorithm. The code is parallel and is designed for CUDA GPUs.
It has been tested only on linux machines, the portability on Windows or MacOS is not guaranteed.

## Compilation with MakeFile

Inside the Makefile is possible to compile:

* <code>make fwb_host</code>
    - Floyd Warshall sequential on CPU, both classic and blocked versions.

* <code>make fwb_dev VERSION=\<version\></code>
    - Floyd Warshall parallel on GPU. Since we developed many difference versions with difference performance, is mandatory to specify which version you want to compile. The first compilation will take some time since it has to compile all the objects of the various cpp files. The compilation of the remaining, if desired, is fast since only the cpp of the version still needs to be compiled.

    NOTE: Use underscore instead of poin between version digits, as we did on file names. Example (to compile version 2.1):
    <code>make fwb_dev VERSION=2_1</code>

* <code>make generate_n_b</code>
    - Generates a list of couples (n,B) accordingly the parameters. Useful for correctness and efficiency testing purposes. 

## Binary Execution

The binaries are generated inside the <code>bin</code> directory.

Both host and device Floyd Warshall binaries share the same parameters, which are handled in the file <code>handle_arguments_and_execute.cu</code>.
It is possible also to pass "--help" to read the guide.
The parameters are the following:

* <code>exec_option=\<test\|perf\|\launch></code>.

    - <code>launch</code>: just executes the matrix given as input. Additional params:
        - (mandatory) <code>--input-file=\<file\></code>: matrix csv input file
        - (mandatory) <code>-b=\<b\></code>: blocking factor

    - <code>perf</code>: executes the algorithm and calculates the performances. It is possible to pass the matrix in input or to generate randomly the matrixes.

        - (mandatory) <code>-b=\<b\></code>: blocking factor
        - (mandatory) <code>-t=\<t\></code>: number of tests
        - (optional)  <code>-n=\<n\></code>: matrix size (mandatory to generate random matrix)
        - (optional)  <code>--input-file=\<file\></code>: matrix csv input file (mandatory for use a csv matrix) 
        - (optional) <code>--output-file=\<file\></code>: csv of all outputs (only in case of analyzer is chrono, with nvprof every couple (n,B) will produce its output csv) (default is csv/chrono_performances.csv)
        - (optional) <code>--analyzer=\<chrono\|nvprof\></code>: the analyzer to use (default is chrono)
        - (optional) <code>-s=\<s\></code>: the seed of the matrix to generate randomly (default is RANDOM, only in case of matrix random generation)


    - <code>test</code>: will execute automatically 500 different tests per each random couple (n,B), comparing the version compiled and the host function. If the two matrixes are not equal, a counter of the number errors is increased and the seed used as input is printed. At the end of each couple the number of errors (new and total) is printed. Since this was designed to check the correctness during the developments part, the parameters used cannot be passed through the terminal. If you desire to see or change their values you can set defaults in the <code>statistical_test.hpp</code> file or customize them inside the <code>handle_arguments_and_execute.cpp</code>.

## Python Compilation and Execution

Instead of launching a single binary we developed two python scripts, one for testing and one for performance, which automatically 