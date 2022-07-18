read_matrix :	
	nvcc -rdc=true -o bin/read_matrix.out \
		main.cpp \
		src/adj_matrix_reader.cpp \
		src/adj_matrix_utils.cpp

fwm:
	g++ -o bin/fwm.out \
		floyd_warshall_matrix.cpp \
		src/adj_matrix_utils.cpp  \
		src/host_floyd_warshall.cpp \
		 

fwa:
	g++ -o bin/fwa.out \
		floyd_warshall_array.cpp \
		src/adj_matrix_utils.cpp \
		src/host_floyd_warshall.cpp

fwa_dev:
	nvcc -o bin/fwa_dev.out \
		floyd_warshall_array_device.cu \
		src/adj_matrix_utils.cpp \
		src/cuda_errors_utils.cu \
		src/host_floyd_warshall.cpp 
	