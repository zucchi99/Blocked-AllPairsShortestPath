#include "adj_matrix_reader.h"

#include <iostream>
#include <string>


using namespace std;

int main() {

	const string adjMatrixFile = "C:\\Users\\Proprietario\\Downloads\\graph_adj_matrix.csv";
	const char delim = ' ';
	
	int numberOfNodes;

	cout << "Lettura file: C:\\Users\\Proprietario\\Downloads\\graph_adj_matrix.csv" << endl;

	int** adjMatrix = readAdjMatrixCSV(adjMatrixFile, delim, &numberOfNodes);

	cout << "Individuata matrice " << numberOfNodes << " x " << numberOfNodes << endl;

	cout << "Stampa matrice:" << endl;

	printAdjMatrix(adjMatrix, numberOfNodes, delim);
}