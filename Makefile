NVCC ?= $(or $(shell which nvcc 2>/dev/null),$(wildcard $(CUDA_HOME)/bin/nvcc),$(wildcard /usr/local/cuda/bin/nvcc),$(wildcard /opt/cuda/bin/nvcc),nvcc)
CFLAGS = -O2 -std=c++20 -arch=native
LDFLAGS = -lcudart -lpthread -lnccl

p2pmark: p2pmark.cu
	$(NVCC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f p2pmark

.PHONY: clean
