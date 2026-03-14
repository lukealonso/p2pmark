#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <thread>
#include <barrier>

#define CHECK(cmd) do { cudaError_t e = cmd; if (e != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while(0)

// Test: one socket-0 GPU writes to multiple socket-1 GPUs simultaneously.
// If each destination routes through a different xGMI link, aggregate BW
// should scale with number of targets.
int main() {
  int ngpu;
  CHECK(cudaGetDeviceCount(&ngpu));
  printf("Found %d GPUs\n\n", ngpu);

  // Enable P2P.
  for (int i = 0; i < ngpu; i++) {
    CHECK(cudaSetDevice(i));
    for (int j = 0; j < ngpu; j++) {
      if (i == j) continue;
      int can;
      CHECK(cudaDeviceCanAccessPeer(&can, i, j));
      if (can) cudaDeviceEnablePeerAccess(j, 0);
    }
  }

  size_t sz = 64 * 1024 * 1024;
  int iters = 100;

  // Socket-1 GPUs (assumed 4,5,6).
  std::vector<int> s1_gpus = {4, 5, 6};
  int src = 0;

  void* src_buf;
  CHECK(cudaSetDevice(src));
  CHECK(cudaMalloc(&src_buf, sz));
  CHECK(cudaMemset(src_buf, 0xAB, sz));

  // Allocate dst bufs on each socket-1 GPU.
  std::vector<void*> dst_bufs(s1_gpus.size());
  for (int r = 0; r < (int)s1_gpus.size(); r++) {
    CHECK(cudaSetDevice(s1_gpus[r]));
    CHECK(cudaMalloc(&dst_bufs[r], sz));
  }

  printf("=== GPU %d writing to cross-socket peers (one at a time) ===\n\n", src);
  CHECK(cudaSetDevice(src));

  for (int r = 0; r < (int)s1_gpus.size(); r++) {
    int peer = s1_gpus[r];
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // Warmup.
    for (int w = 0; w < 10; w++)
      CHECK(cudaMemcpyPeerAsync(dst_bufs[r], peer, src_buf, src, sz, stream));
    CHECK(cudaStreamSynchronize(stream));

    cudaEvent_t e0, e1;
    CHECK(cudaEventCreate(&e0));
    CHECK(cudaEventCreate(&e1));
    CHECK(cudaEventRecord(e0, stream));
    for (int it = 0; it < iters; it++)
      CHECK(cudaMemcpyPeerAsync(dst_bufs[r], peer, src_buf, src, sz, stream));
    CHECK(cudaEventRecord(e1, stream));
    CHECK(cudaStreamSynchronize(stream));
    float ms;
    CHECK(cudaEventElapsedTime(&ms, e0, e1));
    printf("  GPU %d -> GPU %d: %.2f GB/s\n", src, peer, (double)sz * iters / (ms/1000.0) / 1e9);

    CHECK(cudaEventDestroy(e0));
    CHECK(cudaEventDestroy(e1));
    CHECK(cudaStreamDestroy(stream));
  }

  printf("\n=== GPU %d writing to ALL cross-socket peers simultaneously ===\n\n", src);

  int n = s1_gpus.size();
  std::vector<cudaStream_t> streams(n);
  CHECK(cudaSetDevice(src));
  for (int r = 0; r < n; r++)
    CHECK(cudaStreamCreate(&streams[r]));

  // Warmup.
  for (int r = 0; r < n; r++) {
    for (int w = 0; w < 10; w++)
      CHECK(cudaMemcpyPeerAsync(dst_bufs[r], s1_gpus[r], src_buf, src, sz, streams[r]));
  }
  for (int r = 0; r < n; r++)
    CHECK(cudaStreamSynchronize(streams[r]));

  cudaEvent_t e0, e1;
  CHECK(cudaEventCreate(&e0));
  CHECK(cudaEventCreate(&e1));
  CHECK(cudaEventRecord(e0, streams[0]));
  for (int r = 1; r < n; r++)
    CHECK(cudaStreamWaitEvent(streams[r], e0, 0));

  for (int it = 0; it < iters; it++) {
    for (int r = 0; r < n; r++)
      CHECK(cudaMemcpyPeerAsync(dst_bufs[r], s1_gpus[r], src_buf, src, sz, streams[r]));
  }

  for (int r = 1; r < n; r++) {
    cudaEvent_t done;
    CHECK(cudaEventCreate(&done));
    CHECK(cudaEventRecord(done, streams[r]));
    CHECK(cudaStreamWaitEvent(streams[0], done, 0));
    CHECK(cudaEventDestroy(done));
  }
  CHECK(cudaEventRecord(e1, streams[0]));
  CHECK(cudaStreamSynchronize(streams[0]));

  float ms;
  CHECK(cudaEventElapsedTime(&ms, e0, e1));
  double total = (double)sz * n * iters / (ms/1000.0) / 1e9;
  printf("  Aggregate: %.2f GB/s (%.2f per target)\n", total, total / n);
  printf("  If 1 link:  ~46 GB/s expected\n");
  printf("  If 3 links: ~138 GB/s expected\n");

  CHECK(cudaEventDestroy(e0));
  CHECK(cudaEventDestroy(e1));

  // Also test: ALL socket-0 GPUs writing to ALL socket-1 GPUs.
  printf("\n=== ALL socket-0 GPUs writing to ALL socket-1 GPUs simultaneously ===\n\n");

  std::vector<int> s0_gpus = {0, 1, 2, 3};
  // Allocate src/dst per pair.
  std::vector<void*> s0_src(s0_gpus.size());
  std::vector<std::vector<void*>> xs_dst(s0_gpus.size(), std::vector<void*>(s1_gpus.size()));
  std::vector<std::vector<cudaStream_t>> xs_streams(s0_gpus.size(), std::vector<cudaStream_t>(s1_gpus.size()));

  for (int i = 0; i < (int)s0_gpus.size(); i++) {
    CHECK(cudaSetDevice(s0_gpus[i]));
    CHECK(cudaMalloc(&s0_src[i], sz));
    CHECK(cudaMemset(s0_src[i], i, sz));
    for (int j = 0; j < (int)s1_gpus.size(); j++) {
      CHECK(cudaSetDevice(s1_gpus[j]));
      CHECK(cudaMalloc(&xs_dst[i][j], sz));
      CHECK(cudaSetDevice(s0_gpus[i]));
      CHECK(cudaStreamCreate(&xs_streams[i][j]));
    }
  }

  // Warmup.
  for (int i = 0; i < (int)s0_gpus.size(); i++) {
    CHECK(cudaSetDevice(s0_gpus[i]));
    for (int j = 0; j < (int)s1_gpus.size(); j++) {
      for (int w = 0; w < 10; w++)
        CHECK(cudaMemcpyPeerAsync(xs_dst[i][j], s1_gpus[j], s0_src[i], s0_gpus[i], sz, xs_streams[i][j]));
    }
  }
  for (int i = 0; i < (int)s0_gpus.size(); i++) {
    CHECK(cudaSetDevice(s0_gpus[i]));
    for (int j = 0; j < (int)s1_gpus.size(); j++)
      CHECK(cudaStreamSynchronize(xs_streams[i][j]));
  }

  std::barrier xs_bar((int)s0_gpus.size());
  std::vector<double> xs_bw(s0_gpus.size());
  std::vector<std::thread> xs_threads;

  for (int i = 0; i < (int)s0_gpus.size(); i++) {
    xs_threads.emplace_back([&, i]() {
      CHECK(cudaSetDevice(s0_gpus[i]));

      cudaEvent_t start, stop;
      CHECK(cudaEventCreate(&start));
      CHECK(cudaEventCreate(&stop));

      xs_bar.arrive_and_wait();

      CHECK(cudaEventRecord(start, xs_streams[i][0]));
      for (int j = 1; j < (int)s1_gpus.size(); j++)
        CHECK(cudaStreamWaitEvent(xs_streams[i][j], start, 0));

      for (int it = 0; it < iters; it++) {
        for (int j = 0; j < (int)s1_gpus.size(); j++)
          CHECK(cudaMemcpyPeerAsync(xs_dst[i][j], s1_gpus[j], s0_src[i], s0_gpus[i], sz, xs_streams[i][j]));
      }

      for (int j = 1; j < (int)s1_gpus.size(); j++) {
        cudaEvent_t done;
        CHECK(cudaEventCreate(&done));
        CHECK(cudaEventRecord(done, xs_streams[i][j]));
        CHECK(cudaStreamWaitEvent(xs_streams[i][0], done, 0));
        CHECK(cudaEventDestroy(done));
      }
      CHECK(cudaEventRecord(stop, xs_streams[i][0]));
      CHECK(cudaStreamSynchronize(xs_streams[i][0]));

      float ms;
      CHECK(cudaEventElapsedTime(&ms, start, stop));
      xs_bw[i] = (double)sz * s1_gpus.size() * iters / (ms/1000.0) / 1e9;

      CHECK(cudaEventDestroy(start));
      CHECK(cudaEventDestroy(stop));
    });
  }
  for (auto& t : xs_threads) t.join();

  double xs_total = 0;
  for (int i = 0; i < (int)s0_gpus.size(); i++) {
    printf("  GPU %d -> socket 1: %.2f GB/s\n", s0_gpus[i], xs_bw[i]);
    xs_total += xs_bw[i];
  }
  printf("\n  Cross-socket aggregate: %.2f GB/s\n", xs_total);
  printf("  3x xGMI theoretical:   ~200 GB/s\n");

  // Cleanup.
  for (int i = 0; i < (int)s0_gpus.size(); i++) {
    CHECK(cudaSetDevice(s0_gpus[i]));
    CHECK(cudaFree(s0_src[i]));
    for (int j = 0; j < (int)s1_gpus.size(); j++) {
      CHECK(cudaSetDevice(s1_gpus[j]));
      CHECK(cudaFree(xs_dst[i][j]));
      CHECK(cudaSetDevice(s0_gpus[i]));
      CHECK(cudaStreamDestroy(xs_streams[i][j]));
    }
  }

  CHECK(cudaSetDevice(src));
  CHECK(cudaFree(src_buf));
  for (int r = 0; r < n; r++) {
    CHECK(cudaSetDevice(s1_gpus[r]));
    CHECK(cudaFree(dst_bufs[r]));
    CHECK(cudaStreamDestroy(streams[r]));
  }

  return 0;
}
