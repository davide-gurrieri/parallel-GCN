#!/bin/bash
mkdir -p output
mkdir -p output/plot
make tuning-accuracy-no-feature
exec/tuning-accuracy-no-feature
