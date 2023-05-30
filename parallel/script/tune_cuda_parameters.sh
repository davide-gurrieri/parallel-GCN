#!/bin/bash
mkdir -p output
mkdir -p output/plot
# Compile with the right flags
make OPTIONAL_CXXFLAGS="-DDYNAMIC_INPUT -DDEBUG_CUDA -DFEATURE -DTUNE"

for dataset in "cora" "pubmed" "citeseer"
do
    # Name of the output file
    file_name="output/tuning_cuda_$dataset.txt"
    # Delete output file (if exists)
    if [ -e "$file_name" ]; then
        rm "$file_name"
    fi
    # echo "num_blocks_factor num_threads avg_epoch_training_time" >> "$file_name"
    for ((num_blocks_factor=1; num_blocks_factor<=16; num_blocks_factor++))
    do
        for num_threads in 128 256 512 1024
        do
            # Define the content to be written, including the variable value
            content="#-------------------------------------------------
# Parameters for GCN
#-------------------------------------------------
hidden_dim = 16
dropout_input = 0.5
dropout_layer1 = 0.5
epochs = 100
early_stopping = 0

#-------------------------------------------------
# Parameters for Adam
#-------------------------------------------------
learning_rate = 0.01
weight_decay = 5e-4
beta1 = 0.9
beta2 = 0.999
eps = 1e-8

#-------------------------------------------------
# Parameters for Cuda
#-------------------------------------------------
num_blocks_factor = $num_blocks_factor
num_threads = $num_threads"

            # Write the content to the file
            echo "$content" > parameters.txt
            # Run the algorithm
            iter=10
            sum=0
            for ((i=1; i<=$iter; i++))
            do
                time=$(./exec/gcn-par $dataset)
                sum=$(echo "$sum + $time" | bc)
            done
            avg_time=$(echo "scale=3; $sum / $iter" | bc)
            #time=$(./exec/gcn-par $dataset)
            echo "$num_blocks_factor $num_threads $avg_time" >> "$file_name"
        done
    done
    python3 script/ordering.py $file_name
done
