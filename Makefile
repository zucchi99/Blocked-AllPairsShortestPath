main :	
	nvcc -rdc=true main.cpp src/adj_matrix_reader.cpp -o bin/main.out

floyd_warshall_matrix:
	gcc floyd_warshall_matrix.c src/adj_matrix_utils.cpp -o bin/fwm.out