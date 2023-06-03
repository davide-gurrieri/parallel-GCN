CXX=nvcc
CXXFLAGS=-O3 -std=c++17 -rdc=true -arch=sm_75
OPTIONAL_CXXFLAGS=-DDEBUG_CUDA #-DNO_FEATURE #-DTUNE
CXXFILES=src/parser.cpp src/gcn.cu src/variable.cu src/module.cu src/timer.cpp src/sparse.cu src/optim.cu src/reduction.cu src/smart_object.cu
HFILES=include/gcn.cuh include/parser.h include/utils.cuh include/variable.cuh include/module.cuh include/timer.h include/sparse.cuh include/optim.cuh include/shared_ptr.cuh

all: gcn-par

gcn-par: src/main.cpp $(CXXFILES) $(HFILES)
	mkdir -p exec
	$(CXX) $(CXXFLAGS) $(OPTIONAL_CXXFLAGS) -o exec/gcn-par $(CXXFILES) src/main.cpp

gcn-par-no-feature: src/main.cpp $(CXXFILES) $(HFILES)
	mkdir -p exec
	$(CXX) $(CXXFLAGS) -DNO_FEATURE -o exec/gcn-par-no-feature $(CXXFILES) src/main.cpp

gcn-par-tune: src/main.cpp $(CXXFILES) $(HFILES)
	mkdir -p exec
	$(CXX) $(CXXFLAGS) -DTUNE_CUDA -o exec/gcn-par-tune $(CXXFILES) src/main.cpp

clean:
	rm exec/*
