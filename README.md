# Blocked-AllPairsShortestPath

In this repository we propose our implementation for the Blocked Floyd Warshall APSP algorithm. The code is parallel and is designed for CUDA GPUs.
It has been tested only on linux machines, the portability on Windows or MacOS is not guaranteed.

## Repository structure:

Folders:
* .: contains readme.md, python launchers, our report, original paper, make file, python notebook for data analysis, html file which links to the overleaf of the report, .gitignore, changelog.md
* bin: the binaries and the objects compiled (files *.o and *.out)
* csv: contains all csv output (from chrono, nvprof or generate_and_print_n_b)
* input_graphs: instances of csv graphs to import (actually the code works with any folder)
* main: contains the *.cu, *.cpp containing the <code>main</code> function. Which are: the floyd warshall versions and the <code>generate_and_print_n_b.cpp</code> file
* include: contains all the headers *.hpp and *.cuh
* src: contains all the file *.cpp and *cu of the headers specified in the include folder 
* png: contains the images of the plots exported after the data analysis

## Compilation with MakeFile

With the Makefile command is possible to compile:

* <code>make fwb_host</code>
    - Binary path file name: <code>bin/fwb_host.out</code>
    - Floyd Warshall sequential on CPU, both classic and blocked versions.

* <code>make fwb_dev VERSION=\<version\></code>
    - Binary path file name: <code>bin/fwb_dev_v_\<version\>.out</code>
    - Floyd Warshall parallel on GPU. Since we developed many difference versions with difference performance, is mandatory to specify which version you want to compile. The first compilation will take some time since it has to compile all the objects of the various cpp files. The compilation of the remaining, if desired, is fast since only the cpp of the version still needs to be compiled. NB: The version parameter must be equal to one of the files of floyd warshall inside the <code>main</code> directory, for example 2_1.
    - Example: <code>make fwb_dev VERSION=2_1</code> will produce <code>bin/fwb_dev_v_2_1.out</code>.

* <code>make generate_n_b</code>
    - Binary path file name: <code>bin/generate_and_print_n_b.out</code>
    - Generates a list of couples (n,B). Useful for correctness and efficiency testing purposes. 

## Floyd Warshall Binary Execution

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

    Main file: <code>src/performance_test.cu</code>
        
    Example: <code>bin/fwb_dev_v_2_1.out perf --input-file="input_graphs/example_1.csv" -b=2 -t=10</code>
        
    Example: <code>bin/fwb_dev_v_2_1.out perf -n=20 -b=2 -t=10 -s=16362 --output_file="csv/chrono_performances.out" --analyzer=chrono</code>
        
    Example: <code>bin/fwb_dev_v_2_1.out perf -n=20 -b=2 -t=10</code>

    - <code>test</code>: will execute automatically 500 different tests per each random couple (n,B), comparing the version compiled and the host function. If the two matrixes are not equal, a counter of the number errors is increased and the seed used as input is printed. At the end of each couple the number of errors (new and total) is printed. Since this was designed to check the correctness during the developments part, the parameters used cannot be passed through the terminal. If you desire to see or change their values you can set defaults in the <code>statistical_test.hpp</code> file or customize them inside the <code>handle_arguments_and_execute.cpp</code>. 
        
    Main file: <code>src/statistical_test.cpp</code>

    Example: <code>bin/fwb_dev_v_2_1.out test</code>

## Generate and Print (n,b) Binary Execution
Binary file name: <code>bin/generate_and_print_n_b.out</code>
Since this was designed to generate random values during the developments part, not all the parameters can be passed through the terminal. 
Actually all parameters regarding the random generation must be setted in the cpp file.
As default behaviour, the code generates the random couples of (n,b) in this way: <code>next(n) = to_mul * n + to_sum</code>, using a <code>min_n<code> as first n and <code>max_n</code> as upper bound. 
The b are taken randomly between its divisors. There is no control if n obtained is a prime number, in case no b are found the current n is discarded. the to_mul value is a random double between 1.3 and 1.6, the to_sum value is a random integer between 0 and 100.

Example: <code>bin/generate_and_print_n_b.out</code>

It is possible to generate all the possible couples of (n,b) given two lists, one for all n and one for all b.
In this case all couples of (n,b) s.t. b is a divisor of n are kept.

The list of n,b generated is printed to a csv default file: <code>csv/list_of_n_b.csv</code>

Example: <code>bin/generate_and_print_n_b.out 80,160,240,320,480,640,720,1200 8,16,24,32</code>

## Python Launchers Execution

Instead of launching a single binary we developed two python scripts, one for testing and one for performance, which automatically compile and execute all the versions using random matrixes. For the python tests no parameters are needed, for the one of perfomances only the analyzer (chrono, default, or nvprof).

The python for the tests don't generates itself all the values for n and b since they already randomly obtained at the start of the <code>multi_size_statistical_test</code> function in the file <code>statistical_test.cpp</code>.

The python for the performance analysis instead calls the itself the <code>generate_and_print_n_b.cpp</code> generating all couples of (n,b) given the two lists (see the example before).
Then it reads the csv with pandas and iterates over the rows launching the binary with the current n,b values.

We remember that python is an interpreted language, so there is no need of compilation, just invoke directly the python interpreter with the desired parameters.

Examples:
<code>python launch_test.py</code>,
<code>python launch_perf.py chrono</code>,
<code>python launch_perf.py nvprof</code>.

## Data Analysis with Python Notebook
To analyse the execution timings obtained during the performances we used a python notebook.
It automatically generates the dataframe reading the output csv both for chrono and nvprof and prints the plots both in the notebook and to image files in the png folder.
It is easily possible to filter the df to see a detail of some specific versions or matrix sizes, inside the notebook there are two cells where can be specified two lists: <code>versions_to_remove<code> and <code>sizes_to_remove</code>.
