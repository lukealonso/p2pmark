#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

#define CHECK(cmd) do { cudaError_t e = cmd; if (e != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while(0)

// NVIDIA's delay kernel — blocks stream until flag is set.
__global__ void delay(volatile int *flag, unsigned long long timeout_clocks = 10000000) {
  long long int start_clock = clock64();
  while (!*flag) {
    if (clock64() - start_clock > timeout_clocks) break;
  }
}

// Match NVIDIA's p2pBandwidthLatencyTest EXACTLY.
int main() {
  int numGPUs;
  CHECK(cudaGetDeviceCount(&numGPUs));
  printf("GPUs: %d\n\n", numGPUs);

  int numElems = 40000000;  // NVIDIA default: 40M ints = 160 MB
  int repeat = 5;

  volatile int *flag = NULL;
  CHECK(cudaHostAlloc((void**)&flag, sizeof(*flag), cudaHostAllocPortable));

  std::vector<int*> buffers(numGPUs);
  std::vector<cudaEvent_t> start(numGPUs), stop(numGPUs);
  std::vector<cudaStream_t> stream(numGPUs);

  for (int d = 0; d < numGPUs; d++) {
    CHECK(cudaSetDevice(d));
    CHECK(cudaStreamCreateWithFlags(&stream[d], cudaStreamNonBlocking));
    CHECK(cudaMalloc(&buffers[d], numElems * sizeof(int)));
    CHECK(cudaMemset(buffers[d], 0, numElems * sizeof(int)));
    CHECK(cudaEventCreate(&start[d]));
    CHECK(cudaEventCreate(&stop[d]));
  }

  // Enable ALL peer access upfront (our approach).
  for (int i = 0; i < numGPUs; i++) {
    CHECK(cudaSetDevice(i));
    for (int j = 0; j < numGPUs; j++) {
      if (i == j) continue;
      int access;
      CHECK(cudaDeviceCanAccessPeer(&access, i, j));
      if (access) cudaDeviceEnablePeerAccess(j, 0);
    }
  }

  printf("=== P2P Writes, ALL peer access pre-enabled (our approach) ===\n");
  printf("   D\\D");
  for (int j = 0; j < numGPUs; j++) printf("%7d", j);
  printf("\n");

  for (int i = 0; i < numGPUs; i++) {
    CHECK(cudaSetDevice(i));
    printf("%6d ", i);
    for (int j = 0; j < numGPUs; j++) {
      if (i == j) { printf("%7.2f", 0.0); continue; }

      CHECK(cudaStreamSynchronize(stream[i]));
      *flag = 0;
      delay<<<1, 1, 0, stream[i]>>>(flag);
      CHECK(cudaEventRecord(start[i], stream[i]));

      for (int r = 0; r < repeat; r++)
        CHECK(cudaMemcpyPeerAsync(buffers[j], j, buffers[i], i,
                                  sizeof(int) * numElems, stream[i]));

      CHECK(cudaEventRecord(stop[i], stream[i]));
      *flag = 1;
      CHECK(cudaStreamSynchronize(stream[i]));

      float time_ms;
      CHECK(cudaEventElapsedTime(&time_ms, start[i], stop[i]));
      double gb = numElems * sizeof(int) * repeat / 1e9;
      printf("%7.2f", gb / (time_ms / 1e3));
    }
    printf("\n");
  }

  // Disable all peer access.
  for (int i = 0; i < numGPUs; i++) {
    CHECK(cudaSetDevice(i));
    for (int j = 0; j < numGPUs; j++) {
      if (i == j) continue;
      cudaDeviceDisablePeerAccess(j);
    }
  }

  printf("\n=== P2P Writes, enable/disable per-pair (NVIDIA approach) ===\n");
  printf("   D\\D");
  for (int j = 0; j < numGPUs; j++) printf("%7d", j);
  printf("\n");

  for (int i = 0; i < numGPUs; i++) {
    CHECK(cudaSetDevice(i));
    printf("%6d ", i);
    for (int j = 0; j < numGPUs; j++) {
      if (i == j) { printf("%7.2f", 0.0); continue; }

      int access;
      CHECK(cudaDeviceCanAccessPeer(&access, i, j));
      if (access) {
        cudaDeviceEnablePeerAccess(j, 0);
        CHECK(cudaSetDevice(j));
        cudaDeviceEnablePeerAccess(i, 0);
        CHECK(cudaSetDevice(i));
      }

      CHECK(cudaStreamSynchronize(stream[i]));
      *flag = 0;
      delay<<<1, 1, 0, stream[i]>>>(flag);
      CHECK(cudaEventRecord(start[i], stream[i]));

      for (int r = 0; r < repeat; r++)
        CHECK(cudaMemcpyPeerAsync(buffers[j], j, buffers[i], i,
                                  sizeof(int) * numElems, stream[i]));

      CHECK(cudaEventRecord(stop[i], stream[i]));
      *flag = 1;
      CHECK(cudaStreamSynchronize(stream[i]));

      float time_ms;
      CHECK(cudaEventElapsedTime(&time_ms, start[i], stop[i]));
      double gb = numElems * sizeof(int) * repeat / 1e9;
      printf("%7.2f", gb / (time_ms / 1e3));

      if (access) {
        cudaDeviceDisablePeerAccess(j);
        CHECK(cudaSetDevice(j));
        cudaDeviceDisablePeerAccess(i);
        CHECK(cudaSetDevice(i));
      }
    }
    printf("\n");
  }

  for (int d = 0; d < numGPUs; d++) {
    CHECK(cudaSetDevice(d));
    CHECK(cudaFree(buffers[d]));
    CHECK(cudaEventDestroy(start[d]));
    CHECK(cudaEventDestroy(stop[d]));
    CHECK(cudaStreamDestroy(stream[d]));
  }
  CHECK(cudaFreeHost((void*)flag));
  return 0;
}
