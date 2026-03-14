#include <cuda_runtime.h>
#include <cstdio>

#define CHECK(cmd) do { cudaError_t e = cmd; if (e != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while(0)

__global__ void delay(volatile int *flag, unsigned long long timeout_clocks = 10000000) {
  long long int start_clock = clock64();
  while (!*flag) {
    if (clock64() - start_clock > timeout_clocks) break;
  }
}

int main() {
  int src = 0, dst = 4;
  CHECK(cudaSetDevice(src));
  cudaDeviceEnablePeerAccess(dst, 0);
  CHECK(cudaSetDevice(dst));
  cudaDeviceEnablePeerAccess(src, 0);

  size_t max_sz = 256ULL * 1024 * 1024;
  void *src_buf, *dst_buf;
  CHECK(cudaSetDevice(src));
  CHECK(cudaMalloc(&src_buf, max_sz));
  CHECK(cudaMemset(src_buf, 0xAB, max_sz));
  CHECK(cudaSetDevice(dst));
  CHECK(cudaMalloc(&dst_buf, max_sz));

  volatile int *flag;
  CHECK(cudaHostAlloc((void**)&flag, sizeof(*flag), cudaHostAllocPortable));

  cudaStream_t stream;
  CHECK(cudaSetDevice(src));
  CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  cudaEvent_t e0, e1;
  CHECK(cudaEventCreate(&e0));
  CHECK(cudaEventCreate(&e1));

  printf("GPU %d -> GPU %d (cross-socket)\n\n", src, dst);
  printf("%12s  %12s  %12s  %12s\n", "Size", "No delay", "With delay", "Diff");
  printf("------------ ------------ ------------ ------------\n");

  size_t sizes[] = {
    4*1024*1024, 8*1024*1024, 16*1024*1024, 32*1024*1024,
    64*1024*1024, 128*1024*1024, 160*1024*1024, 256*1024*1024
  };

  for (size_t sz : sizes) {
    int reps = (int)(800ULL * 1024 * 1024 / sz);  // ~800 MB total
    if (reps < 3) reps = 3;

    // Warmup.
    for (int w = 0; w < 5; w++)
      CHECK(cudaMemcpyPeerAsync(dst_buf, dst, src_buf, src, sz, stream));
    CHECK(cudaStreamSynchronize(stream));

    // Without delay kernel.
    CHECK(cudaEventRecord(e0, stream));
    for (int r = 0; r < reps; r++)
      CHECK(cudaMemcpyPeerAsync(dst_buf, dst, src_buf, src, sz, stream));
    CHECK(cudaEventRecord(e1, stream));
    CHECK(cudaStreamSynchronize(stream));
    float ms1;
    CHECK(cudaEventElapsedTime(&ms1, e0, e1));
    double gbps_no_delay = (double)sz * reps / (ms1 / 1000.0) / 1e9;

    // With delay kernel (NVIDIA style).
    *flag = 0;
    delay<<<1, 1, 0, stream>>>(flag);
    CHECK(cudaEventRecord(e0, stream));
    for (int r = 0; r < reps; r++)
      CHECK(cudaMemcpyPeerAsync(dst_buf, dst, src_buf, src, sz, stream));
    CHECK(cudaEventRecord(e1, stream));
    *flag = 1;
    CHECK(cudaStreamSynchronize(stream));
    float ms2;
    CHECK(cudaEventElapsedTime(&ms2, e0, e1));
    double gbps_delay = (double)sz * reps / (ms2 / 1000.0) / 1e9;

    char sz_str[32];
    if (sz >= 1024*1024) snprintf(sz_str, sizeof(sz_str), "%zu MB", sz / (1024*1024));
    printf("%12s  %10.2f    %10.2f    %+.1f%%\n", sz_str, gbps_no_delay, gbps_delay,
           (gbps_delay - gbps_no_delay) / gbps_no_delay * 100);
  }

  CHECK(cudaEventDestroy(e0));
  CHECK(cudaEventDestroy(e1));
  CHECK(cudaStreamDestroy(stream));
  CHECK(cudaSetDevice(src));
  CHECK(cudaFree(src_buf));
  CHECK(cudaSetDevice(dst));
  CHECK(cudaFree(dst_buf));
  CHECK(cudaFreeHost((void*)flag));
  return 0;
}
