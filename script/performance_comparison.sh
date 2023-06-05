#!/bin/bash
# from principal directory
mkdir -p output
mkdir -p output/plot
cd hpdga-spring23/
make performance-cpu
cd ..
make performance-gpu
./exec/performance-cpu
./exec/performance-gpu
