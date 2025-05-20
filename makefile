/*

For more information refer to this code 

https://github.com/NVIDIA/cuda-samples

*/

CUDA_HOME ?= /usr/local/cuda

all: graph.x

graph.x: graph.cu gpu_graph.hpp gpu_graph.cpp cuda_helper.hpp
	$(CUDA_HOME)/bin/nvcc -g -lineinfo -gencode arch=compute_70,code=sm_70 graph.cu gpu_graph.cpp -o graph.x -I$(CUDA_HOME)/include -L$(CUDA_HOME)/lib64 -lcudart
