NVCC ?= $(or $(shell which nvcc 2>/dev/null),$(wildcard $(CUDA_HOME)/bin/nvcc),$(wildcard /usr/local/cuda/bin/nvcc),$(wildcard /opt/cuda/bin/nvcc),nvcc)
CFLAGS = -O2 -std=c++20 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_100,code=sm_100 -gencode arch=compute_120,code=sm_120
LDFLAGS = -lcudart -lpthread -lnccl

p2pmark: p2pmark.cu
	$(NVCC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f p2pmark

.PHONY: clean
