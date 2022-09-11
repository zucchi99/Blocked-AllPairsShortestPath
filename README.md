# Blocked-AllPairsShortestPath

In this repository we propose our implementation for the Blocked Floyd Warshall APSP algorithm. The code is parallel and is designed for CUDA GPUs.
It has been tested only on linux machines, the portability on Windows or MacOS is not guaranteed.

## Compilation with MakeFile

Inside the Makefile is possible to compile:

* <code>make fwb_host</code>
    - Floyd Warshall sequential on CPU, both classic and blocked versions.

* <code>make fwb_dev VERSION=\<version\></code>
    - Floyd Warshall parallel on GPU. Since we developed many difference versions with difference performance, is mandatory to specify which version you want to compile. The first compilation will take some time since it has to compile all the objects of the various cpp files. The compilation of the remaining, if desired, is fast since only the cpp of the version still needs to be compiled.

* <code>make generate_n_b</code>
    - Generates a list of couples (n,B) accordingly the parameters. Useful for correctness and efficiency testing purposes. 

## Execution

The binaries are generated inside the <code>bin</code> directory.

Both host and device Floyd Warshall binaries share the same parameters, which are handled in the file <code>host_floyd_warshall.cpp</code>.
It is possible also to pass "--help" to read the guide.
The parameters are the following:

* <code>exec_option</code> which can be <code>test</code> or <code>perf</code>.

    - <code>launch</code>: executes the matrix given as input


    - <code>perf</code>: executes the algorithm performances, in s


    - <code>test</code>: will execute automatically 500 different tests per each random couple (n,B), comparing the version compiled and the host function. If the two matrixes are not equal, a counter of the number errors is increased and the seed used as input is printed. At the end of each couple the number of errors (new and total) is printed. Since this was designed to check the correctness during the developments part, the parameters used cannot be passed through the terminal. If you desire to see or change their values you can set defaults in the <code>statistical_test.hpp</code> file or customize them inside the <code>handle_arguments_and_execute.cpp</code>.



