# parallel-GCN

High-performance CUDA C++ implementation of a Graph Convolutional Network

# Description

This repository is developed as part of the "High-Performance Data and Graph Analytics" contest at Politecnico di Milano (see the [contest repository](https://github.com/ian-ofgod/hpdga-spring23)). It provides a CUDA implementation of a parallelized Graph Convolutional Network (GCN), that is tuned to maximize the classification accuracy of some popular datasets.

The parallelism enhance training and inference speed, reaching a speedup of up to 15x compared to the sequential version. Moreover, classification accuracy is improved by generalizing the number of layers and doing optimal parameters tuning.

# Prerequisites

To compile the code it is necessary to install a complete cuda environment. For example:

```bash
# Download and install CUDA 12.1
set -x \
&& cd $(mktemp -d) \
&& wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run \
&& sudo sh cuda_12.1.0_530.30.02_linux.run --silent --toolkit \
&& rm cuda_12.1.0_530.30.02_linux.run
```

To execute some script you need `bc`

```bash
sudo apt install bc
```

# Installation

```bash
git clone https://github.com/davide-gurrieri/parallel-GCN.git
cd parallel-GCN/parallel/
```

# Compilation

```bash

```
