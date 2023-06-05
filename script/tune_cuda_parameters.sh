#!/bin/bash
mkdir -p output
mkdir -p output/plot
make tuning-cuda
exec/tuning-cuda
python3 script/ordering.py output/tuning_cuda_citeseer.txt
python3 script/ordering.py output/tuning_cuda_cora.txt
python3 script/ordering.py output/tuning_cuda_pubmed.txt
python3 script/ordering.py output/tuning_cuda_reddit.txt
