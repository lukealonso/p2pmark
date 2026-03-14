#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

#define CHECK(cmd) do { cudaError_t e = cmd; if (e != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while(0)

// Test whether buffer allocation pattern affects cross-socket bandwidth.
int main() {
  int ngpu;
  CHECK(cudaGetDeviceCount(&ngpu));

  for (int i = 0; i < ngpu; i++) {
    CHECK(cudaSetDevice(i));
    for (int j = 0; j < ngpu; j++) {
      if (i == j) continue;
      int can; CHECK(cudaDeviceCanAccessPeer(&can, i, j));
      if (can) cudaDeviceEnablePeerAccess(j, 0);
    }
  }

  int src = 0, dst = 4;
  size_t sz = 64 * 1024 * 1024;
  int reps = 20;

  auto bench = [&](void* s, void* d) {
    cudaStream_t stream;
    CHECK(cudaSetDevice(src));
    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    for (int w = 0; w < 5; w++)
      CHECK(cudaMemcpyPeerAsync(d, dst, s, src, sz, stream));
    CHECK(cudaStreamSynchronize(stream));
    cudaEvent_t e0, e1;
    CHECK(cudaEventCreate(&e0));
    CHECK(cudaEventCreate(&e1));
    CHECK(cudaEventRecord(e0, stream));
    for (int r = 0; r < reps; r++)
      CHECK(cudaMemcpyPeerAsync(d, dst, s, src, sz, stream));
    CHECK(cudaEventRecord(e1, stream));
    CHECK(cudaStreamSynchronize(stream));
    float ms;
    CHECK(cudaEventElapsedTime(&ms, e0, e1));
    CHECK(cudaEventDestroy(e0));
    CHECK(cudaEventDestroy(e1));
    CHECK(cudaStreamDestroy(stream));
    return (double)sz * reps / (ms / 1000.0) / 1e9;
  };

  // Method 1: allocate only on src and dst (like our measure_bw).
  {
    void *s, *d;
    CHECK(cudaSetDevice(src));
    CHECK(cudaMalloc(&s, sz));
    CHECK(cudaMemset(s, 0xAB, sz));
    CHECK(cudaSetDevice(dst));
    CHECK(cudaMalloc(&d, sz));
    double bw = bench(s, d);
    printf("Alloc only src+dst:     %.2f GB/s\n", bw);
    CHECK(cudaSetDevice(src)); CHECK(cudaFree(s));
    CHECK(cudaSetDevice(dst)); CHECK(cudaFree(d));
  }

  // Method 2: allocate on ALL GPUs first, then test (like NVIDIA/xgmi_match).
  {
    std::vector<void*> bufs(ngpu);
    for (int i = 0; i < ngpu; i++) {
      CHECK(cudaSetDevice(i));
      CHECK(cudaMalloc(&bufs[i], sz));
      CHECK(cudaMemset(bufs[i], i, sz));
    }
    double bw = bench(bufs[src], bufs[dst]);
    printf("Alloc on ALL GPUs:      %.2f GB/s\n", bw);
    for (int i = 0; i < ngpu; i++) {
      CHECK(cudaSetDevice(i));
      CHECK(cudaFree(bufs[i]));
    }
  }

  // Method 3: allocate on src and dst, but also allocate dummy bufs on other GPUs.
  {
    std::vector<void*> dummies(ngpu);
    for (int i = 0; i < ngpu; i++) {
      if (i == src || i == dst) continue;
      CHECK(cudaSetDevice(i));
      CHECK(cudaMalloc(&dummies[i], sz));
    }
    void *s, *d;
    CHECK(cudaSetDevice(src));
    CHECK(cudaMalloc(&s, sz));
    CHECK(cudaMemset(s, 0xAB, sz));
    CHECK(cudaSetDevice(dst));
    CHECK(cudaMalloc(&d, sz));
    double bw = bench(s, d);
    printf("Alloc src+dst+dummies:  %.2f GB/s\n", bw);
    CHECK(cudaSetDevice(src)); CHECK(cudaFree(s));
    CHECK(cudaSetDevice(dst)); CHECK(cudaFree(d));
    for (int i = 0; i < ngpu; i++) {
      if (i == src || i == dst) continue;
      CHECK(cudaSetDevice(i));
      CHECK(cudaFree(dummies[i]));
    }
  }

  // Method 4: allocate LARGE buffer on dst, test write to different offsets.
  {
    void *s;
    CHECK(cudaSetDevice(src));
    CHECK(cudaMalloc(&s, sz));
    CHECK(cudaMemset(s, 0xAB, sz));

    size_t big = 2ULL * 1024 * 1024 * 1024;  // 2 GB
    void *d;
    CHECK(cudaSetDevice(dst));
    CHECK(cudaMalloc(&d, big));

    printf("\nLarge dst buffer, writes to different offsets:\n");
    for (size_t off = 0; off < big; off += 256 * 1024 * 1024) {
      double bw = bench(s, (char*)d + off);
      printf("  offset %4zu MB: %.2f GB/s\n", off / (1024*1024), bw);
    }

    CHECK(cudaSetDevice(src)); CHECK(cudaFree(s));
    CHECK(cudaSetDevice(dst)); CHECK(cudaFree(d));
  }

  return 0;
}
