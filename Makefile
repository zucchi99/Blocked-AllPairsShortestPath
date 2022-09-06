#compilare con:
#make dev VERSION=1_1

#compilers
NVCC=nvcc
GXX=g++

#directories
src_path=src
header_path=include

# floyd warshall blocked - cuda parallelism
fwb_dev:
	@echo "version:" $(VERSION)
	nvcc -rdc=true -o bin/fwa_dev_v_$(VERSION).out \
		device_floyd_warshall_v_$(VERSION).cu \
		src/adj_matrix_utils.cpp \
		src/cuda_errors_utils.cu \
		src/generate_n_b_couples.cpp \
		src/handle_arguments_and_execute.cu \
		src/host_floyd_warshall.cpp \
		src/math.cpp \
		src/performance_test.cu \
		src/statistical_test.cpp

# floyd warshall blocked - sequential
fwb_host:
	g++ -o bin/fwa.out \
		host_floyd_warshall_array.cpp \
		src/adj_matrix_utils.cpp \
		src/host_floyd_warshall.cpp

# generates list of (n,B), for testing purposes
generate_n_b:
	g++ -o bin/generate_and_print_n_b.out \
		generate_and_print_n_b.cpp \
		src/generate_n_b_couples.cpp