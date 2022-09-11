# Blocked-AllPairsShortestPath

In this repository we propose our implementation for the Blocked Floyd Warshall APSP algorithm. The code is parallel and is designed for CUDA GPUs.
It has been tested only on linux machines, the portability on Windows or MacOS is not guaranteed.

## Compilation with MakeFile

Inside the Makefile is possible to compile:

* <code>make fwb_host</code>
    - Floyd Warshall sequential on CPU, both classic and blocked versions

* <code>make fwb_dev VERSION=\<version\></code>
    - Floyd Warshall parallel on GPU. Since we developed many difference versions with difference performance, is mandatory to specify which version you want to compile. The first compilation will take some time since it has to compile all the objects of the various cpp files. The compilation of the remaining, if desired, is fast since only the cpp of the version still needs to be compiled.

* <code>make generate_n_b</code>
    - Generates a list of couples (n,B) accordingly the parameters. Useful fo correctness and efficiency testing purposes. 

## Execution
