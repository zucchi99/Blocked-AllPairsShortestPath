import re #regex
import os #operating system
import sys
import pandas as pd


# INPUT TEST DIMENSIONS

#define and print command for compiling test dimension cpp file
make_test_dim_cmd = "make generate_n_b"
print(make_test_dim_cmd)
# compile test dimension cpp file to bin file
os.system(make_test_dim_cmd)
print()

# execute test dimension bin file
os.system("bin/generate_and_print_n_b.out")

test_dimensions = pd.read_csv('csv/list_of_n_b.csv')
print("test_dimensions:")
print(test_dimensions)
print()

# -----------------------------------------------------------------

# ALGORTIHM EXECUTION

cuda_files = os.listdir()
#print(files)

for file in cuda_files :

    if re.match("device_floyd_warshall_v_.*\.cu", file) and (not re.match("device_floyd_warshall_v_.*_ERROR.cu", file)):

        # is a floy_warshall cuda file

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

            launch_cmd = "nvprof --csv --log-file " + csv_output + " --normalized-time-unit us --profile-from-start off ./" + fw_bin + " " + exec_option + " -t=" + str(t) + " -n=" + str(n) + " -b=" + str(b)
            print(launch_cmd)
            os.system(launch_cmd)
            print()

        print("-----------------------------------------------------------------")
        print()
