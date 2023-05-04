CXX=nvcc
CXXFLAGS= -O3 -std=c++17
LDFLAGS=-lm -lstdc++

CXXFILES=src/gcn.cpp src/parser.cpp
HFILES=include/gcn.h include/parser.h

all: gcn-par

gcn-par: src/main.cpp $(CXXFILES) $(HFILES)
	mkdir -p exec
	$(CXX) $(CXXFLAGS) -o exec/gcn-par $(CXXFILES) src/main.cpp $(LDFLAGS)

clean:
	rm exec/*

