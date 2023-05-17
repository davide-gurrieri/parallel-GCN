CXX=nvcc
CXXFLAGS= -O3 -std=c++17
CXXFILES=src/parser.cpp src/gcn.cu src/variable.cu src/module.cu src/timer.cpp src/sparse.cu src/optim.cu src/reduction.cu
HFILES=include/gcn.cuh include/parser.h include/utils.cuh include/variable.cuh include/module.cuh include/timer.h include/sparse.cuh include/optim.cuh include/shared_ptr.cuh

all: gcn-par

gcn-par: src/main.cpp $(CXXFILES) $(HFILES)
	mkdir -p exec
	$(CXX) $(CXXFLAGS) -o exec/gcn-par $(CXXFILES) src/main.cpp

clean:
	rm exec/*
