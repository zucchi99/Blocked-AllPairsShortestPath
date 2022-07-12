#include "adj_matrix_reader.h"
#include <fstream>
#include <sstream>

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

		adjMatrix[lineNumber][i] = std::stoi(itemStr);
		i++;
	}

	return -1;
}


int readAdjMatrixCSV(const std::string filename, const char delim, int** adjMatrix, int *numberOfNodes) {

	std::ifstream fs(filename);
	
	if (!fs.is_open()) {
		return -1;
	}
	
	if (!fs.eof()) {
		return -1;
	}

	// read first line
	std::string line;
	std::getline(fs, line);
	int lineNumber = 0;
	
	// get number of nodes
	*numberOfNodes = _getNumberOfNodes(line, delim);

	// allocate memory for matrix
	adjMatrix = (int**) malloc((*numberOfNodes) * (*numberOfNodes) * sizeof(int));

	// parse all lines and fill adjMatrix
	do {
		_parseLine(line, delim, lineNumber, adjMatrix);
		lineNumber++;
	} while (std::getline(fs, line));

	return 0;
}



