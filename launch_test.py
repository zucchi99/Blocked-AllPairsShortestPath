import re #regex
import os #operating system
import pandas as pd
import random
import sys

#default all
versions_to_test = []
for i in range(1, len(sys.argv)) :
    versions_to_test.append(sys.argv[i])
        
files = os.listdir()
cuda_files = [ file for file in files if re.match("device_floyd_warshall_v_.*\.cu", file) and (not re.match("device_floyd_warshall_v_.*_ERROR\.cu", file)) ]
file_i = 1
num_files = len(cuda_files)

for file in cuda_files :

    # obtain cuda file version
    version = re.sub("^device_floyd_warshall_v_", "", file)
    version = re.sub("\.cu$", "", version)
    
    if (versions_to_test == []) or (version in versions_to_test) :

        # define floyd warshall bin file
        fw_bin = 'bin/fwa_dev_v_' + version + '.out'
        
        # print version, cuda file name, bin file name
        print(f"file:      {file_i} of {num_files}")
        print(f"version:   {version}")
        print(f"cuda file: {file}")
        print(f"bin  file: {fw_bin}\n")

        # define and print command for compiling cuda file to bin file
        make_algorithm_cmd = "make dev VERSION=" + version
        print(make_algorithm_cmd)
        # compile test dimension cpp file to bin file
        os.system(make_algorithm_cmd)
        print()

        exec_option='test'

        launch_cmd = fw_bin + " " + exec_option
            
        print(launch_cmd, "\n")
        os.system(launch_cmd)            
        
        print("-----------------------------------------------------------------")
        print()
    
    file_i += 1
