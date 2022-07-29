%%file Makefile
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

fwa_dev_v_1_0:
	nvcc -o bin/fwa_dev_v_1_0.out \
		floyd_warshall_device_v_1_0.cu \
		src/adj_matrix_utils.cu \
		src/adj_matrix_utils.cpp \
		src/cuda_errors_utils.cu \
		src/performance_test.cu \
		src/statistical_test.cpp \
		src/host_floyd_warshall.cpp 

fwa_dev_v_1_1:
	nvcc -o bin/fwa_dev_v_1_1.out \
		floyd_warshall_device_v_1_1.cu \
		src/adj_matrix_utils.cu \
		src/adj_matrix_utils.cpp \
		src/cuda_errors_utils.cu \
		src/performance_test.cu \
		src/statistical_test.cpp \
		src/host_floyd_warshall.cpp 

fwa_dev_v_1_2:
	nvcc -o bin/fwa_dev_v_1_2.out \
		floyd_warshall_device_v_1_2.cu \
		src/adj_matrix_utils.cu \
		src/adj_matrix_utils.cpp \
		src/cuda_errors_utils.cu \
		src/performance_test.cu \
		src/statistical_test.cpp \
		src/host_floyd_warshall.cpp
		
fwa_dev_v_1_3:
	nvcc -o bin/fwa_dev_v_1_3.out \
		floyd_warshall_device_v_1_3.cu \
		src/adj_matrix_utils.cu \
		src/adj_matrix_utils.cpp \
		src/cuda_errors_utils.cu \
		src/performance_test.cu \
		src/statistical_test.cpp \
		src/host_floyd_warshall.cpp
	

fwa_dev_pitch:
	nvcc -o bin/fwa_dev_pitch.out \
		floyd_warshall_device_v_pitch.cu \
		src/adj_matrix_utils.cu \
		src/adj_matrix_utils.cpp \
		src/cuda_errors_utils.cu \
		src/performance_test.cu \
		src/statistical_test.cpp \
		src/host_floyd_warshall.cpp