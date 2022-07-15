#ifndef ADJ_MATRIX_READER_HPP
#define ADJ_MATRIX_READER_HPP

#include <string>

/// <summary>
/// Leggi un file CSV contenente una matrice di adiacenza.
/// </summary>
/// <param name="filename">Il nome del file da leggere</param>
/// <param name="delim">Il delimitatore del file CSV</param>
/// <param name="adjMatrix">Un puntatore vuoto dove verr� allocata la memoria per mem. la matrice</param>
/// <param name="numberOfNodes">Il numero di nodi del grafo (corrisponde al numero di righe e colonne della matrice)</param>
/// <return>0 if okay, else something less</return>
int** readAdjMatrixCSV(std::string filename, const char delim, int* numberOfNodes);

/// <summary>
/// Stampa a schermo una matrice di adiacenza (utile a fini di debug).
/// </summary>
/// <param name="adjMatrix"></param>
/// <param name="numberOfNodes"></param>
/// <param name="delim"></param>
/// <returns></returns>
void printAdjMatrix(int** adjMatrix, int numberOfNodes, const char delim);

#endif
