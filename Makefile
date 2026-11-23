CXX ?= g++
NVCC ?= nvcc
CXXFLAGS ?= -std=c++17 -O3 -g -ffp-contract=off -Wall -Wextra -Wpedantic
OMPFLAGS ?= -fopenmp
CUDA_ARCH ?= sm_75
NVCCFLAGS ?= -std=c++17 -O3 -lineinfo --fmad=false -arch=$(CUDA_ARCH)


.PHONY: all cpu test clean
all: cpu
cpu: build/legacy build/barenn_cpu

build:
	mkdir -p build

build/legacy: barenn.cpp | build
	$(CXX) $(CXXFLAGS) $< -o $@

build/barenn_cpu: src/main.cpp src/core.hpp barenn.cpp | build
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) src/main.cpp -o $@

build/validate: tests/validate.cpp src/core.hpp barenn.cpp | build
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) tests/validate.cpp -o $@

test: build/validate build/barenn_cpu
	./build/validate
	python3 -m unittest discover -s tests -p 'test_*.py'

clean:
	rm -f build/legacy build/barenn_cpu build/validate
