
#include <stdio.h>
#include <stdlib.h>


#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

#include "../include/adj_matrix_utils.hpp"

// ---------------------------------------------------------------
//  PRINT UTILS

void print_matrix(int **matrix, int m, int n) {
    printf("[\n");
    for (int i = 0; i < m; i++) {
        printf("  ");
        print_array(matrix[i], n);
    }
    printf("]\n");
}

void print_array(int *array, int size) {
    printf("[");
    for (int i = 0; i < size; i++) {
        print_element(array[i], INF);
        if (i < size-1) printf(", ");
    }
    printf("]\n");
}

void print_element(int val, int infinity) {
    if (val < infinity)
        printf("%02d", val);
    else 
        printf("--");
}

// ---------------------------------------------------------------
// MATRIX GENERATION, COMPARE and others utils

bool same_matrixes(int **matrix_1, int **matrix_2, int m, int n, bool oriented_graph) {
    for (int i = 0; i < m; i++) {
        for (int j = oriented_graph ? 0 : (i+1); j < n; j++) {
            if(matrix_1[i][j] != matrix_2[i][j]) return false;
        }
    }
    return true;
}

int** allocate_matrix(int m, int n) {
    //matrix with row major order:
    //m rows pointers
    int** matrix = (int **) malloc(sizeof(int *) * m);
    for (int i = 0; i < m; i++) {
        //each row has n intengers
        matrix[i] = (int *) malloc(sizeof(int) * n);
    }
    return matrix;
}

void populate_adj_matrix(int **matrix, int n, int seed, bool oriented_graph) {
    for (int i = 0; i < n; i++) {
        //diagonal always zero (distance 0 to myself)
        matrix[i][i] = 0;
        for (int j = oriented_graph ? 0 : (i+1); j < n; j++) {
            if (i != j) {               
                bool add_edge = (rand() % 100) <= DENSITY;
                int val = (rand() % MAX_COST) + MIN_COST;
                matrix[i][j] = add_edge ? val : INF;
                if (! oriented_graph) {
                    //non-oriented graph
                    matrix[j][i] = matrix[i][j];
                }        
            }
        }
    }
}

// ---------------------------------------------------------------
// ARRAY MATRIX FUNCTIONS VARIANTS

void copy_arr_matrix(int *dest_matrix, int *source_matrix, int m, int n) {
    // dest_matrix = allocate_arr_matrix(m, n);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            dest_matrix[i*n + j] = source_matrix[i*n + j];
        }
    }
}

void print_arr_matrix(int *matrix, int m, int n) {
    printf("[\n");
    for (int i = 0; i < m; i++) {
        printf("  ");
        print_array(&(matrix[i*n]), n);
    }
    printf("]\n");
}

bool same_arr_matrixes(int *matrix_1, int *matrix_2, int m, int n, bool oriented_graph) {
    for (int i = 0; i < m; i++) {
        for (int j = oriented_graph ? 0 : (i+1); j < n; j++) {
            if(matrix_1[i*n + j] != matrix_2[i*n + j]) return false;
        }
    }
    return true;
}

int* allocate_arr_matrix(int m, int n) {
    return (int *) malloc(sizeof(int) * m * n);
}

void populate_arr_adj_matrix(int* arr_matrix, int n, int seed, bool oriented_graph) {
    for (int i = 0; i < n; i++) {
        arr_matrix[i*n + i] = 0;
        for (int j = oriented_graph ? 0 : (i+1); j < n; j++) {
            if (i != j) {        
                bool add_edge = (rand() % 100) <= DENSITY;
                int val = (rand() % MAX_COST) + MIN_COST;
                arr_matrix[i*n + j] = add_edge ? val : INF;
                if (! oriented_graph) {
                    //non-oriented graph
                    arr_matrix[j*n + i] = arr_matrix[i*n + j];
                }
            }
        }
    }
}

int _getNumberOfNodes(std::string adjMatrixLine, const char delim) {

	// insipired to: https://java2blog.com/split-string-space-cpp/#Using_getline_Method

	std::istringstream ss(adjMatrixLine);

	int nodesCounter = 0;

	std::string s;
	while (std::getline(ss, s, delim)) {
		nodesCounter++;
	}

	return nodesCounter;
}

int _parseLine(std::string lineAsStr, const char delim, int lineNumber, int* matrix, int nNodes) {

	// insipired to: https://java2blog.com/split-string-space-cpp/#Using_getline_Method

	std::istringstream ss(lineAsStr);
	std::string itemStr;

	int i = 0;
	while (std::getline(ss, itemStr, delim)) {

		int value = std::stoi(itemStr);
		matrix[lineNumber*nNodes+i] = value;

		i++;
	}

	return 0;
}


int* read_arr_matrix(int* numberOfNodes, std::string filename, const char delim) {

	std::ifstream fs(filename);
	
	if (!fs.is_open()) {
		// todo: add error
	}
	
	if (fs.eof()) {
		// todo: add error
	}

	// read first line
	std::string line;
	std::getline(fs, line);
	int lineNumber = 0;
	
	// get number of nodes
	*numberOfNodes = _getNumberOfNodes(line, delim);

    int* matrix = allocate_arr_matrix(*numberOfNodes, *numberOfNodes);

    printf("Detected %d nodes reading matrix.\n", *numberOfNodes);

	// parse all lines and fill adjMatrix
	do {
		_parseLine(line, delim, lineNumber, matrix, *numberOfNodes);
		lineNumber++;
	} while (std::getline(fs, line));

    return matrix;
}
