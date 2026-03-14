#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>
#include <vector>

#define CHECK(cmd) do { cudaError_t e = cmd; if (e != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while(0)

// Test cross-socket P2P write bandwidth at different transfer sizes.
// Helps find the xGMI interleaving granularity.
int main() {
  int src = 0, dst = 4;  // cross-socket pair
  printf("Testing GPU %d -> GPU %d (cross-socket) at various transfer sizes\n\n", src, dst);

  CHECK(cudaSetDevice(src));
  cudaDeviceEnablePeerAccess(dst, 0);
  CHECK(cudaSetDevice(dst));
  cudaDeviceEnablePeerAccess(src, 0);

  // Allocate a large buffer on each side.
  size_t max_buf = 1ULL * 1024 * 1024 * 1024;  // 1 GB
  void *src_buf, *dst_buf;
  CHECK(cudaSetDevice(src));
  CHECK(cudaMalloc(&src_buf, max_buf));
  CHECK(cudaMemset(src_buf, 0xAB, max_buf));
  CHECK(cudaSetDevice(dst));
  CHECK(cudaMalloc(&dst_buf, max_buf));

  cudaStream_t stream;
  CHECK(cudaSetDevice(src));
  CHECK(cudaStreamCreate(&stream));

  printf("%12s  %10s  %10s\n", "Size", "GB/s", "Iters");
  printf("------------ ---------- ----------\n");

  std::vector<size_t> sizes = {
    1*1024*1024, 2*1024*1024, 4*1024*1024, 8*1024*1024,
    16*1024*1024, 32*1024*1024, 64*1024*1024, 128*1024*1024,
    256*1024*1024, 512*1024*1024, 1024*1024*1024ULL
  };

  for (size_t sz : sizes) {
    int iters = std::max(10, (int)(4ULL * 1024 * 1024 * 1024 / sz));  // ~4 GB total

    // Warmup.
    for (int w = 0; w < 5; w++)
      CHECK(cudaMemcpyPeerAsync(dst_buf, dst, src_buf, src, sz, stream));
    CHECK(cudaStreamSynchronize(stream));

    cudaEvent_t e0, e1;
    CHECK(cudaEventCreate(&e0));
    CHECK(cudaEventCreate(&e1));
    CHECK(cudaEventRecord(e0, stream));
    for (int i = 0; i < iters; i++)
      CHECK(cudaMemcpyPeerAsync(dst_buf, dst, src_buf, src, sz, stream));
    CHECK(cudaEventRecord(e1, stream));
    CHECK(cudaStreamSynchronize(stream));

    float ms;
    CHECK(cudaEventElapsedTime(&ms, e0, e1));
    double gbps = (double)sz * iters / (ms / 1000.0) / 1e9;

    char sz_str[32];
    if (sz >= 1024*1024*1024ULL) snprintf(sz_str, sizeof(sz_str), "%zu GB", sz / (1024*1024*1024));
    else snprintf(sz_str, sizeof(sz_str), "%zu MB", sz / (1024*1024));
    printf("%12s  %10.2f  %10d\n", sz_str, gbps, iters);

    CHECK(cudaEventDestroy(e0));
    CHECK(cudaEventDestroy(e1));
  }

  // Now test with MULTIPLE concurrent streams to different offsets.
  printf("\n--- Multiple concurrent streams from different buffer regions ---\n");
  printf("%12s  %10s  %10s\n", "Streams", "GB/s total", "Per-stream");
  printf("------------ ---------- ----------\n");

  for (int nstreams = 1; nstreams <= 16; nstreams *= 2) {
    size_t per_stream_sz = max_buf / nstreams;
    int iters = std::max(10, (int)(4ULL * 1024 * 1024 * 1024 / max_buf));

    std::vector<cudaStream_t> streams(nstreams);
    for (int s = 0; s < nstreams; s++)
      CHECK(cudaStreamCreate(&streams[s]));

    // Warmup.
    for (int s = 0; s < nstreams; s++) {
      size_t off = s * per_stream_sz;
      CHECK(cudaMemcpyPeerAsync((char*)dst_buf + off, dst, (char*)src_buf + off, src,
                                per_stream_sz, streams[s]));
    }
    for (int s = 0; s < nstreams; s++)
      CHECK(cudaStreamSynchronize(streams[s]));

    cudaEvent_t e0, e1;
    CHECK(cudaEventCreate(&e0));
    CHECK(cudaEventCreate(&e1));
    CHECK(cudaEventRecord(e0, streams[0]));
    for (int s = 1; s < nstreams; s++)
      CHECK(cudaStreamWaitEvent(streams[s], e0, 0));

    for (int it = 0; it < iters; it++) {
      for (int s = 0; s < nstreams; s++) {
        size_t off = s * per_stream_sz;
        CHECK(cudaMemcpyPeerAsync((char*)dst_buf + off, dst, (char*)src_buf + off, src,
                                  per_stream_sz, streams[s]));
      }
    }

    for (int s = 1; s < nstreams; s++) {
      cudaEvent_t done;
      CHECK(cudaEventCreate(&done));
      CHECK(cudaEventRecord(done, streams[s]));
      CHECK(cudaStreamWaitEvent(streams[0], done, 0));
      CHECK(cudaEventDestroy(done));
    }
    CHECK(cudaEventRecord(e1, streams[0]));
    CHECK(cudaStreamSynchronize(streams[0]));

    float ms;
    CHECK(cudaEventElapsedTime(&ms, e0, e1));
    double gbps = (double)max_buf * iters / (ms / 1000.0) / 1e9;
    printf("%12d  %10.2f  %10.2f\n", nstreams, gbps, gbps / nstreams);

    CHECK(cudaEventDestroy(e0));
    CHECK(cudaEventDestroy(e1));
    for (int s = 0; s < nstreams; s++)
      CHECK(cudaStreamDestroy(streams[s]));
  }

  CHECK(cudaSetDevice(src));
  CHECK(cudaFree(src_buf));
  CHECK(cudaStreamDestroy(stream));
  CHECK(cudaSetDevice(dst));
  CHECK(cudaFree(dst_buf));
  return 0;
}
