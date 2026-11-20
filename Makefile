CXX ?= g++
NVCC ?= nvcc
CXXFLAGS ?= -std=c++17 -O3 -g -ffp-contract=off -Wall -Wextra -Wpedantic
OMPFLAGS ?= -fopenmp
NVCCFLAGS ?= -std=c++17 -O3 -lineinfo --fmad=false

.PHONY: all cpu cuda test clean
all: cpu
cpu: build/legacy

build:
	mkdir -p build

build/legacy: barenn.cpp | build
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -f build/legacy build/barenn_cpu build/barenn_cuda build/validate
