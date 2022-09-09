import re #regex
import os #operating system
import sys

for file in os.listdir() :
    new_file = ""
    if (re.match(".*v_1_1", file)) :
        new_file = re.sub("v_1_1", "v_1_0", file)
    elif (re.match(".*v_1_2", file)) :
        new_file = re.sub("v_1_2", "v_1_1", file)
    elif (re.match(".*v_1_3", file)) :
        new_file = re.sub("v_1_3", "v_1_2", file)
    elif (re.match(".*v_1_4", file)) :
        new_file = re.sub("v_1_4", "v_1_3", file)
    elif (re.match(".*v_2_2", file)) :
        new_file = re.sub("v_2_2", "v_2_1", file)
    elif (re.match(".*v_3_1", file)) :
        new_file = re.sub("v_3_1", "v_4_0", file)
    elif (re.match(".*v_3_2", file)) :
        new_file = re.sub("v_3_2", "v_4_1", file)
    
    if new_file != "" :
        print(file, " -->", new_file)
        cmd = "mv " + file + " " + new_file
        print(cmd)
        os.system(cmd)
