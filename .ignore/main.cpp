#include <iostream>
#include <string>

#include "include/adj_matrix_reader.hpp"
#include "include/adj_matrix_utils.hpp"

using namespace std;

int main() {

	const string adjMatrixFile = "./graphs_istances/easy_graph.csv";
	const char delim = ' ';
	
	int numberOfNodes;

	cout << "Lettura file: " << endl;

	int** adjMatrix = readAdjMatrixCSV(adjMatrixFile, delim, &numberOfNodes);

	cout << "Individuata matrice " << numberOfNodes << " x " << numberOfNodes << endl;

	cout << "Stampa matrice:" << endl;

	print_matrix(adjMatrix, numberOfNodes, numberOfNodes);
}