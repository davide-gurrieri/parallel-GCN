#!/bin/bash
mkdir -p output
mkdir -p output/plot
make tuning-accuracy
exec/tuning-accuracy
