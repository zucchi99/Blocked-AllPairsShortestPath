import re #regex
import os #operating system
import pandas as pd
import random

# INPUT TEST DIMENSIONS

# define and print command for compiling test dimension cpp file
make_test_dim_cmd = "make generate_n_b"
print(make_test_dim_cmd)
# compile test dimension cpp file to bin file
os.system(make_test_dim_cmd)
print()

# generate all test dimensions
os.system("bin/generate_and_print_n_b.out 8 16 24 32")
print()

# store test dimensions to csv and print them
test_dimensions = pd.read_csv('csv/list_of_n_b.csv')
print("test_dimensions:")
print(test_dimensions)
print()

# -----------------------------------------------------------------

# ALGORITHM EXECUTION

cuda_files = os.listdir()
#print(files)

# calculate and print the list of all random seeds
all_seeds = []
print("seeds:")
for i in range(len(test_dimensions)) :
    rand = random.randint(0,9999999)
    all_seeds.append(rand)
    print(i, rand)
print()

# test each version
for file in cuda_files :

    if re.match("device_floyd_warshall_v_.*\.cu", file) and (not re.match("device_floyd_warshall_v_.*_ERROR\.cu", file)):

        # is a floyd_warshall cuda file

        # obtain cuda file version
        version = re.sub("^device_floyd_warshall_v_", "", file)
        version = re.sub("\.cu$", "", version)
        
        # define floyd warshall bin file
        fw_bin = 'bin/fwa_dev_v_' + version + '.out'
        
        # print version, cuda file name, bin file name
        print(f"version:   {version}")
        print(f"cuda file: {file}")
        print(f"bin file:  {fw_bin}\n")

        # define and print command for compiling cuda file to bin file
        make_algorithm_cmd = "make dev VERSION=" + version
        print(make_algorithm_cmd)
        # compile test dimension cpp file to bin file
        os.system(make_algorithm_cmd)
        print()

        # define parameters
        #  - exec option: {'perf', 'test'}
        exec_option='perf'
        #  - number of tests for each couple (n,b)
        t=1

        for row in test_dimensions.iterrows() :

            # foreach test dimension couple (n,b) :

            i = row[0]
            n = row[1][0]
            b = row[1][1]
            #print(n, b)
            
            csv_output = 'csv/fwa_dev_v_' + str(version) + '__n_' + str(n).zfill(3) + '__b_' + str(b).zfill(2) + "__t_" + str(t).zfill(2) + ".csv"
            print(f"out file {i:02}: {csv_output}")

            launch_cmd = "nvprof --csv --log-file " + csv_output + " --normalized-time-unit us --profile-from-start off ./" + fw_bin + " " + exec_option + " -t=" + str(t) + " -n=" + str(n) + " -b=" + str(b) + " -s=" + str(all_seeds[i])
            print(launch_cmd)
            os.system(launch_cmd)
            print()

        print("-----------------------------------------------------------------")
        print()
