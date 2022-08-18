version=$1

#compile floyd warshall
make dev VERSION="$version"

echo

#compile generation
make generate_n_b

echo

#run generation
bin/generate_and_print_n_b.out

echo

#read file generated
n_b_file='csv/list_of_n_b.csv'

#define floyd warshall bin file
fw_bin='bin/device_floyd_warshall_v_'"$version"'.out'

#define parameters
exec_option='perf'
t=10

#row index
i=0

#define csv output file name
csv_output='csv/fwa_dev_v_'"$version"'__'$(date "+%Y_%m_%d-%H_%M")'__'
echo "$csv_output"

while read line; do

    #skip row header
    if test $i -gt 0
    then
        n=`echo "$line" | sed s/,.*//`
        b=`echo "$line" | sed s/.*,//`

        cur_csv_out="$csv_output"'n'$n'__b'$b'.csv'

        #echo n: $n, b: $b
        # options:
        #   --csv                     ==> print as csv style
        #   --log-file <file>         ==> output to <file>
        #   --normalized-time-unit us ==> print all data with specified time unit (us = microseconds)
        #   --profile-from-start off  ==> disable profiler at start, profiles only the code inside the cudaProfilerStart() and cudaProfilerEnd() (by default is on)
        nvprof --csv --log-file "$cur_csv_out" --normalized-time-unit us --profile-from-start off ./$fw_bin "$exec_option" -t="$t" -n="$n" -b="$b"

    fi

    i=$(($i+1))


done < $n_b_file