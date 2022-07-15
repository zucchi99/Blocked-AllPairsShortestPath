#include "include/adj_matrix_utils.hpp"

#include <iostream>
#include <string>

using namespace std;

int main() {

	const string adjMatrixFile = "./graphs_istances/graph_random_basic.csv";
	const char delim = ' ';
	
	int numberOfNodes;

	cout << "Lettura file: " << endl;

	int** adjMatrix = readAdjMatrixCSV(adjMatrixFile, delim, &numberOfNodes);

	cout << "Individuata matrice " << numberOfNodes << " x " << numberOfNodes << endl;

	cout << "Stampa matrice:" << endl;

	printAdjMatrix(adjMatrix, numberOfNodes, delim);
}