
# --------------------------------------------------------------------
# VARIABLES

#compilers
NVCC=nvcc
CXX=g++

#directories
dir_src=src
dir_header=include
dir_bin=bin
dir_main=main

#flags
NVCC_FLAGS = -rdc=true

#objects
OBJS = \
	$(dir_bin)/adj_matrix_utils.o \
	$(dir_bin)/cuda_errors_utils.o \
	$(dir_bin)/generate_n_b_couples.o \
	$(dir_bin)/handle_arguments_and_execute.o \
	$(dir_bin)/host_floyd_warshall.o \
	$(dir_bin)/math.o \
	$(dir_bin)/performance_test.o \
	$(dir_bin)/statistical_test.o

OBJ_GEN = $(dir_bin)/generate_n_b_couples.o

# --------------------------------------------------------------------
# MAIN

# floyd warshall blocked - cuda parallelism
# compile with --> make fwb_dev VERSION=1_1
# @echo "version:" $(VERSION)
fwb_dev: $(OBJS)
	$(NVCC) $(NVCC_FLAGS) $(dir_main)/floyd_warshall_device_v_$(VERSION).cu $(OBJS) -o $(dir_bin)/fwb_dev_v_$(VERSION).out
	
# floyd warshall blocked - sequential
fwb_host: $(OBJS)
	$(NVCC) $(NVCC_FLAGS) $(dir_main)/floyd_warshall_host.cu $(OBJS) -o $(dir_bin)/fwb_host.out

# generates list of (n,B), for testing purposes
generate_n_b: $(OBJ_GEN)
	$(CXX) $(dir_main)/generate_and_print_n_b.cpp $(OBJ_GEN) -o bin/generate_and_print_n_b.out

# --------------------------------------------------------------------
# OBJECTS

$(dir_bin)/adj_matrix_utils.o: \
		$(dir_header)/adj_matrix_utils.hpp \
		$(dir_src)/adj_matrix_utils.cpp
	$(CXX) -c $(dir_src)/adj_matrix_utils.cpp -o $(dir_bin)/adj_matrix_utils.o
	
$(dir_bin)/cuda_errors_utils.o: \
		$(dir_header)/cuda_errors_utils.cuh \
		$(dir_src)/cuda_errors_utils.cu
	$(NVCC) -c $(dir_src)/cuda_errors_utils.cu -o $(dir_bin)/cuda_errors_utils.o
	
$(dir_bin)/generate_n_b_couples.o: \
		$(dir_header)/generate_n_b_couples.hpp \
		$(dir_header)/macros.hpp \
		$(dir_src)/generate_n_b_couples.cpp
	$(CXX) -c $(dir_src)/generate_n_b_couples.cpp -o $(dir_bin)/generate_n_b_couples.o
	
$(dir_bin)/handle_arguments_and_execute.o: \
		$(dir_header)/macros.hpp \
		$(dir_bin)/adj_matrix_utils.o \
		$(dir_bin)/cuda_errors_utils.o \
		$(dir_bin)/host_floyd_warshall.o \
		$(dir_bin)/performance_test.o \
		$(dir_bin)/statistical_test.o \
		$(dir_src)/handle_arguments_and_execute.cu
	$(NVCC) -c $(dir_src)/handle_arguments_and_execute.cu -o $(dir_bin)/handle_arguments_and_execute.o

$(dir_bin)/host_floyd_warshall.o: \
		$(dir_header)/host_floyd_warshall.hpp \
		$(dir_bin)/adj_matrix_utils.o \
		$(dir_src)/host_floyd_warshall.cpp
	$(CXX) -c $(dir_src)/host_floyd_warshall.cpp -o $(dir_bin)/host_floyd_warshall.o

$(dir_bin)/math.o: \
		$(dir_header)/math.hpp \
		$(dir_src)/math.cpp
	$(CXX) -c $(dir_src)/math.cpp -o $(dir_bin)/math.o

$(dir_bin)/performance_test.o: \
		$(dir_header)/performance_test.cuh \
		$(dir_bin)/adj_matrix_utils.o \
		$(dir_bin)/math.o \
		$(dir_src)/performance_test.cu
	$(NVCC) -c $(dir_src)/performance_test.cu -o $(dir_bin)/performance_test.o

$(dir_bin)/statistical_test.o: \
		$(dir_header)/statistical_test.hpp \
		$(dir_bin)/adj_matrix_utils.o \
		$(dir_bin)/generate_n_b_couples.o \
		$(dir_bin)/host_floyd_warshall.o \
		$(dir_src)/statistical_test.cpp
	$(CXX) -c $(dir_src)/statistical_test.cpp -o $(dir_bin)/statistical_test.o
