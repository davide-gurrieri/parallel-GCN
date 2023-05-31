#!/bin/bash
# from parallel directory
make OPTIONAL_CXXFLAGS="-DDYNAMIC_INPUT -DDEBUG_CUDA -DFEATURE -DTUNE"
cd ../hpdga-spring23/
make OPTIONAL_CXXFLAGS="-DEVAL"
cd ../parallel/

# Name of the output file
file_name1="output/performance_cpu.txt"
# Delete output file (if exists)
if [ -e "$file_name1" ]; then
    rm "$file_name1"
fi
# Name of the output file
file_name2="output/performance_gpu.txt"
# Delete output file (if exists)
if [ -e "$file_name2" ]; then
    rm "$file_name2"
fi

echo "time dataset" >> "$file_name1"
echo "time dataset" >> "$file_name2"

iter=10
for ((i=1; i<=$iter; i++))
do
    for dataset in "cora" "pubmed" "citeseer"
    do
        time1=$(../hpdga-spring23/exec/gcn-seq $dataset)
        time2=$(./exec/gcn-par $dataset)
        echo "$time1 $dataset" >> "$file_name1"
        echo "$time2 $dataset" >> "$file_name2"
    done
done
