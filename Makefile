
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

$(dir_bin)/adj_matrix_utils.o:
	$(CXX) -c $(dir_src)/adj_matrix_utils.cpp -o $(dir_bin)/adj_matrix_utils.o
	
$(dir_bin)/cuda_errors_utils.o: 
	$(NVCC) -c $(dir_src)/cuda_errors_utils.cu -o $(dir_bin)/cuda_errors_utils.o
	
$(dir_bin)/generate_n_b_couples.o: 
	$(CXX) -c $(dir_src)/generate_n_b_couples.cpp -o $(dir_bin)/generate_n_b_couples.o
	
$(dir_bin)/handle_arguments_and_execute.o:
	$(NVCC) -c $(dir_src)/handle_arguments_and_execute.cu -o $(dir_bin)/handle_arguments_and_execute.o

$(dir_bin)/host_floyd_warshall.o:
	$(CXX) -c $(dir_src)/host_floyd_warshall.cpp -o $(dir_bin)/host_floyd_warshall.o

$(dir_bin)/math.o: 
	$(CXX) -c $(dir_src)/math.cpp -o $(dir_bin)/math.o

$(dir_bin)/performance_test.o:
	$(NVCC) -c $(dir_src)/performance_test.cu -o $(dir_bin)/performance_test.o

$(dir_bin)/statistical_test.o:
	$(CXX) -c $(dir_src)/statistical_test.cpp -o $(dir_bin)/statistical_test.o
