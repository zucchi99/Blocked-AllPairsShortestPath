#compilare con:
#make dev VERSION=1_1

#compilers
NVCC=nvcc
GXX=g++

#directories
src_path=src
header_path=include

#headers
dev:
	@echo "version:" $(VERSION)
	nvcc -o bin/fwa_dev_v_$(VERSION).out \
		device_floyd_warshall_v_$(VERSION).cu \
		src/adj_matrix_utils.cu \
		src/adj_matrix_utils.cpp \
		src/cuda_errors_utils.cu \
		src/performance_test.cu \
		src/statistical_test.cpp \
		src/host_floyd_warshall.cpp 


read_matrix :	
	nvcc -rdc=true -o bin/read_matrix.out \
		main.cpp \
		src/adj_matrix_reader.cpp \
		src/adj_matrix_utils.cpp

fwm:
	g++ -o bin/fwm.out \
		host_floyd_warshall_matrix.cpp \
		src/adj_matrix_utils.cpp  \
		src/host_floyd_warshall.cpp \

fwa:
	g++ -o bin/fwa.out \
		host_floyd_warshall_array.cpp \
		src/adj_matrix_utils.cpp \
		src/host_floyd_warshall.cpp
