# parallel-GCN

High-performance CUDA C++ implementation of Graph Convolutional Networks

## Description

This repository is developed as part of the "High-Performance Data and Graph Analytics" contest at "Politecnico di Milano" university (see the [contest repository](https://github.com/ian-ofgod/hpdga-spring23)). It offers a CUDA implementation of a parallelized Graph Convolutional Network (GCN), which has been optimized to maximize classification accuracy on popular datasets.

The parallelism enhances training and inference speed, resulting in a significant speedup compared to the sequential version. Moreover, classification accuracy is improved by adjusting the number of layers and optimizing the choice of hyperparameters through a validation dataset.

## Prerequisites

To compile the code it is necessary to install a complete cuda environment. For example:

```bash
# Download and install CUDA 12.1
set -x \
&& cd $(mktemp -d) \
&& wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run \
&& sudo sh cuda_12.1.0_530.30.02_linux.run --silent --toolkit \
&& rm cuda_12.1.0_530.30.02_linux.run
```

## Installation

```bash
git clone https://github.com/davide-gurrieri/parallel-GCN.git
cd parallel-GCN/
```

## Compilation and execution

### Part 1: parallel acceleration

Performance evaluation is done using the original parameters of the sequential version to ensure a fair comparison:

```
GCN PARAMETERS:
n_layers = 2
hidden_dim = 16
dropout = 0.5
weight_decay = 5e-4
epochs = 100
early_stopping = 0

ADAM PARAMETERS:
learning_rate = 0.01
beta1 = 0.9
beta2 = 0.999
eps = 1e-8
```

Choose one `dataset_name` from [`citeseer` `cora` `pubmed` `reddit`].

For a single run:

```bash
make
./exec/gcn-par dataset_name
```

To obtain the average time of multiple executions of all datasets:

(200 runs for `citeseer`, `cora` and `pubmed`; 20 runs for `reddit`)

```bash
make performance-gpu
./exec/performance-gpu
```

### Part 2: accuracy improvement

The selected parameters are stored in the [parameters](https://github.com/davide-gurrieri/parallel-GCN/tree/main/parameters) folder.

To run the version optimized for validation accuracy:

```bash
make gcn-par-improvement
make run-dataset_name
```

## Tuning reproducibility

### CUDA parameters selection

To obtain the cuda parameters (number of threads in a block and number of blocks) that minimize the execution time for part 1:

```bash
script/tune_cuda_parameters.sh
```

### GCN parameters selection

To obtain the list of configurations from which the GCN parameters that maximize accuracy are selected:

```bash
# first exploration
make tuning-accuracy
make tuning-accuracy-no-feature
./exec/tuning-accuracy
./exec/tuning-accuracy-no-feature

# second exploration
make tuning-accuracy-second
./exec/tuning-accuracy-second
./exec/tuning-accuracy-second-no-feature
```
