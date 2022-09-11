# Blocked-AllPairsShortestPath

In this repository we propose our implementation for the Blocked Floyd Warshall APSP algorithm. The code is parallel and is designed for CUDA GPUs.
It has been tested only on linux machines, the portability on Windows or MacOS is not guaranteed.

## Compilation with MakeFile

Inside the Makefile is possible to compile:

* <code>make fwb_host</code>
    - Floyd Warshall sequential on CPU, both classic and blocked versions.
    - Binary path file name: <code>bin/fwb_host.out</code>

* <code>make fwb_dev VERSION=\<version\></code>
    - Floyd Warshall parallel on GPU. Since we developed many difference versions with difference performance, is mandatory to specify which version you want to compile. The first compilation will take some time since it has to compile all the objects of the various cpp files. The compilation of the remaining, if desired, is fast since only the cpp of the version still needs to be compiled. The version parameter must be of the files in the <code>main</code> directory, for example 2_1 (note: use underscore)
    - Binary path file name: <code>bin/fwb_dev_v_\<version\>.out</code>
    - Example: <code>make fwb_dev VERSION=2_1</code> will produce <code>bin/fwb_dev_v_2_1.out</code>.

* <code>make generate_n_b</code>
    - Generates a list of couples (n,B) accordingly the parameters. Useful for correctness and efficiency testing purposes. 
    - Binary path file name: <code>bin/generate_and_print_n_b.out</code>

## Binary Execution

The binaries are generated inside the <code>bin</code> directory.

Both host and device Floyd Warshall binaries share the same parameters, which are handled in the file <code>handle_arguments_and_execute.cu</code>.
It is possible also to pass "--help" to read the guide.
The parameters are the following:

* only one is only mandatory: <code>exec_option=\<test\|perf\|launch></code>.

    - <code>launch</code>: just executes the matrix given as input. Additional params:
        - (mandatory) <code>--input-file=\<file\></code>: matrix csv input file
        - (mandatory) <code>-b=\<b\></code>: blocking factor
        
    Example: <code>bin/fwb_dev_v_2_1.out launch --input-file="input_graphs/example_1.csv" -b=2</code>

    - <code>perf</code>: executes the algorithm and calculates the performances. It is possible to pass the matrix in input or to generate randomly the matrixes. Numbers of tests is useful to execute many different times. If random matrixes are used then each test will use a different input matrix otherwise always the input csv one. Is suggested to use t >= 10 to have consistency in the data. Also, in case of chrono, The execution is repeated 20*t times. 20 timings are saved. The execution time returned is the mean of the 20 values. These 20 different times allows to calculate the variancy of the timings.

        - (mandatory) <code>-b=\<b\></code>: blocking factor
        - (mandatory) <code>-t=\<t\></code>: number of tests
        - (optional)  <code>-n=\<n\></code>: matrix size (mandatory to generate random matrix)
        - (optional)  <code>--input-file=\<file\></code>: matrix csv input file (mandatory for use a csv matrix) 
        - (optional)  <code>--output-file=\<file\></code>: csv of all outputs (only in case of analyzer is chrono, with nvprof every couple (n,B) will produce its output csv) (default is csv/chrono_performances.csv)
        - (optional)  <code>--analyzer=\<chrono\|nvprof\></code>: the analyzer to use (default is chrono)
        - (optional)  <code>-s=\<s\></code>: the seed of the matrix to generate randomly (default is RANDOM, only in case of matrix random generation)
        
    Example: <code>bin/fwb_dev_v_2_1.out perf --input-file="input_graphs/example_1.csv" -b=2 -t=10</code>
        
    Example: <code>bin/fwb_dev_v_2_1.out perf -n=20 -b=2 -t=10 -s=16362 --output_file="csv/chrono_performances.out" --analyzer=chrono</code>
        
    Example: <code>bin/fwb_dev_v_2_1.out perf -n=20 -b=2 -t=10</code>

    - <code>test</code>: will execute automatically 500 different tests per each random couple (n,B), comparing the version compiled and the host function. If the two matrixes are not equal, a counter of the number errors is increased and the seed used as input is printed. At the end of each couple the number of errors (new and total) is printed. Since this was designed to check the correctness during the developments part, the parameters used cannot be passed through the terminal. If you desire to see or change their values you can set defaults in the <code>statistical_test.hpp</code> file or customize them inside the <code>handle_arguments_and_execute.cpp</code>.
        
    Example: <code>bin/fwb_dev_v_2_1.out test</code>

## Python Compilation and Execution

Instead of launching a single binary we developed two python scripts, one for testing and one for performance, which automatically compile and execute all the versions using random matrixes. For the python tests no parameters are needed, for the one of perfomances only the analyzer (chrono, default, or nvprof).

Examples:
<code>python launch_test.py</code>,
<code>python launch_perf.py chrono</code>,
<code>python launch_perf.py nvprof</code>.