#include <string>

#pragma once

/// <summary>
/// Leggi un file CSV contenente una matrice di adiacenza.
/// </summary>
/// <param name="filename">Il nome del file da leggere</param>
/// <param name="delim">Il delimitatore del file CSV</param>
/// <param name="adjMatrix">Un puntatore vuoto dove verrà allocata la memoria per mem. la matrice</param>
/// <param name="numberOfNodes">Il numero di nodi del grafo (corrisponde al numero di righe e colonne della matrice)</param>
/// <return>0 if okay, else something less</return>
int readAdjMatrixCSV(std::string filename, std::string delim, int** adjMatrix, int* numberOfNodes);
