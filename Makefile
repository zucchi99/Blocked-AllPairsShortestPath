main :	
	nvcc -rdc=true main.cpp src/adj_matrix_reader.cpp -o bin/main.out

fwm:
	g++ floyd_warshall_matrix.c src/adj_matrix_utils.cpp -o bin/fwm.out

fwa:
	g++ floyd_warshall_array.c src/adj_matrix_utils.cpp -o bin/fwa.out

fwa_dev:
	nvcc fw_array_device.cu src/adj_matrix_utils.cpp -o bin/fwa_dev.out
	