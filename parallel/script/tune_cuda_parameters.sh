#!/bin/bash

# Name of the dataset
dataset="pubmed"

# Name of the output file
file_name="tuning_cuda_$dataset.txt"

# Delete output file (if exists)
if [ -e "$file_name" ]; then
    rm "$file_name"
fi

# Compile with the right flags
make OPTIONAL_CXXFLAGS="-DDYNAMIC_INPUT -DDEBUG_CUDA -DFEATURE -DTUNE"

for ((num_blocks_factor=1; num_blocks_factor<=16; num_blocks_factor++))
do
  for num_threads in 16 32 64 128 256 512 1024
  do
      # Define the content to be written, including the variable value
      content="#-------------------------------------------------
# Parameters for GCN
#-------------------------------------------------
hidden_dim = 16
dropout_input = 0.5
dropout_layer1 = 0.5
epochs = 1000
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
      time=$(./exec/gcn-par $dataset)
      echo "num_blocks_factor=$num_blocks_factor-num_threads=$num_threads $time" >> "$file_name"
  done
done
