#include "../include/adj_matrix_reader.hpp"
#include <fstream>
#include <sstream>
#include <iostream>

//#define out

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

int _parseLine(std::string adjMatrixLine, const char delim, int lineNumber, int** adjMatrix) {

	// insipired to: https://java2blog.com/split-string-space-cpp/#Using_getline_Method

	std::istringstream ss(adjMatrixLine);
	std::string itemStr;

	int i = 0;
	while (std::getline(ss, itemStr, delim)) {

		int value = std::stoi(itemStr);
		adjMatrix[lineNumber][i] = value;

		i++;
	}

	return 0;
}


int** readAdjMatrixCSV(const std::string filename, const char delim, int *numberOfNodes) {

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

	// allocate memory for matrix
	int** adjMatrix = (int **) malloc(sizeof(int*) * (*numberOfNodes));

	// parse all lines and fill adjMatrix
	do {
		adjMatrix[lineNumber] = (int*) malloc(sizeof(int) * (*numberOfNodes));
		_parseLine(line, delim, lineNumber, adjMatrix);
		lineNumber++;
	} while (std::getline(fs, line));

	return adjMatrix;
}


void printAdjMatrix(int** adjMatrix, int numberOfNodes, const char delim) {
	
	for (int i = 0; i < numberOfNodes; i++) {
		for (int j = 0; j < numberOfNodes-1; j++) {
			std::cout << adjMatrix[i][j] << delim;
		}

		std::cout << adjMatrix[i][numberOfNodes - 1] << "\n";
	}
}

