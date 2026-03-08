#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <thread>
#include <barrier>
#include <chrono>

#define CHECK(cmd)                                                        \
  do {                                                                    \
    cudaError_t e = cmd;                                                  \
    if (e != cudaSuccess) {                                               \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
              cudaGetErrorString(e));                                      \
      exit(1);                                                            \
    }                                                                     \
  } while (0)

// Bandwidth mode constants.
static constexpr size_t TRANSFER_SIZE = 64 * 1024 * 1024;  // 64 MB
static constexpr int WARMUP = 5;
static constexpr int ITERS = 20;

// Latency mode constants.
static constexpr size_t LATENCY_SIZE = 128;  // one cacheline
static constexpr int LATENCY_WORDS = LATENCY_SIZE / sizeof(int);
static constexpr int LATENCY_WARMUP = 500;
static constexpr int LATENCY_ITERS = 10000;

// Single-thread kernel that reads `words` ints from a remote GPU pointer.
// P2P access must be enabled so `remote` (allocated on another GPU) is
// directly accessible from the launching GPU.
__global__ void latency_read_kernel(const int* __restrict__ remote, int* local, int words) {
  int sum = 0;
  for (int i = 0; i < words; i++)
    sum += remote[i];
  local[0] = sum;
}

// Measure unidirectional D2D bandwidth from src GPU to dst GPU.
double measure_bw(int src, int dst, size_t bytes, cudaStream_t stream) {
  void *src_buf, *dst_buf;
  CHECK(cudaSetDevice(src));
  CHECK(cudaMalloc(&src_buf, bytes));
  CHECK(cudaMemset(src_buf, 0xAB, bytes));
  CHECK(cudaSetDevice(dst));
  CHECK(cudaMalloc(&dst_buf, bytes));

  CHECK(cudaSetDevice(src));
  for (int i = 0; i < WARMUP; i++)
    CHECK(cudaMemcpyPeerAsync(dst_buf, dst, src_buf, src, bytes, stream));
  CHECK(cudaStreamSynchronize(stream));

  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < ITERS; i++)
    CHECK(cudaMemcpyPeerAsync(dst_buf, dst, src_buf, src, bytes, stream));
  CHECK(cudaStreamSynchronize(stream));
  auto t1 = std::chrono::high_resolution_clock::now();

  double sec = std::chrono::duration<double>(t1 - t0).count();
  double gbps = (double)bytes * ITERS / sec / 1e9;

  CHECK(cudaSetDevice(src));
  CHECK(cudaFree(src_buf));
  CHECK(cudaSetDevice(dst));
  CHECK(cudaFree(dst_buf));

  return gbps;
}

static void run_latency_tests(int ngpu);

int main(int argc, char** argv) {
  bool latency_mode = false;
  for (int i = 1; i < argc; i++) {
    if (std::string(argv[i]) == "--latency") latency_mode = true;
  }

  int ngpu;
  CHECK(cudaGetDeviceCount(&ngpu));
  printf("Found %d GPUs\n\n", ngpu);

  // Enable P2P access between all pairs.
  for (int i = 0; i < ngpu; i++) {
    CHECK(cudaSetDevice(i));
    for (int j = 0; j < ngpu; j++) {
      if (i == j) continue;
      int can;
      CHECK(cudaDeviceCanAccessPeer(&can, i, j));
      if (can) {
        cudaDeviceEnablePeerAccess(j, 0);  // ignore already-enabled error
      }
    }
  }

  if (latency_mode) {
    run_latency_tests(ngpu);
    return 0;
  }

  // ---- Test 1: Sequential NxN ----
  printf("=== Sequential P2P bandwidth (GB/s) ===\n");
  printf("Each transfer: %zu MB, %d iterations\n\n", TRANSFER_SIZE / (1024 * 1024), ITERS);

  std::vector<std::vector<double>> seq_bw(ngpu, std::vector<double>(ngpu, 0.0));

  printf("%6s", "Dst->");
  for (int d = 0; d < ngpu; d++) printf("  GPU%-4d", d);
  printf("\n");

  for (int s = 0; s < ngpu; s++) {
    printf("GPU %d:", s);
    CHECK(cudaSetDevice(s));
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    for (int d = 0; d < ngpu; d++) {
      if (s == d) {
        printf("     -   ");
        continue;
      }
      double bw = measure_bw(s, d, TRANSFER_SIZE, stream);
      seq_bw[s][d] = bw;
      printf("  %6.2f  ", bw);
    }
    printf("\n");
    CHECK(cudaStreamDestroy(stream));
  }

  // ---- Test 2: Staggered circular reads (allreduce pattern) ----
  // N-1 rounds. In round r, every GPU i reads from GPU (i+r+1)%N.
  // This is the access pattern used by the PCIe allreduce kernel to
  // spread traffic across the switch fabric.
  printf("\n=== Topology probe: staggered reads by peer distance (GB/s) ===\n");
  printf("%d concurrent transfers per round, each GPU reading from one unique peer.\n", ngpu);
  printf("Reveals PCIe switch topology: nearby peers are fast, cross-switch peers are slow.\n\n");

  std::vector<void*> stag_bufs(ngpu);
  std::vector<void*> stag_dst(ngpu);
  std::vector<cudaStream_t> stag_streams(ngpu);
  for (int i = 0; i < ngpu; i++) {
    CHECK(cudaSetDevice(i));
    CHECK(cudaMalloc(&stag_bufs[i], TRANSFER_SIZE));
    CHECK(cudaMemset(stag_bufs[i], i, TRANSFER_SIZE));
    CHECK(cudaMalloc(&stag_dst[i], TRANSFER_SIZE));
    CHECK(cudaStreamCreate(&stag_streams[i]));
  }

  for (int r = 0; r < ngpu - 1; r++) {
    int offset = r + 1;
    // Warmup.
    for (int i = 0; i < ngpu; i++) {
      int peer = (i + offset) % ngpu;
      CHECK(cudaSetDevice(i));
      for (int w = 0; w < WARMUP; w++)
        CHECK(cudaMemcpyPeerAsync(stag_dst[i], i, stag_bufs[peer], peer,
                                  TRANSFER_SIZE, stag_streams[i]));
    }
    for (int i = 0; i < ngpu; i++) {
      CHECK(cudaSetDevice(i));
      CHECK(cudaStreamSynchronize(stag_streams[i]));
    }

    // Timed: all GPUs launch simultaneously via events.
    // Record start events.
    std::vector<cudaEvent_t> starts(ngpu), stops(ngpu);
    for (int i = 0; i < ngpu; i++) {
      CHECK(cudaSetDevice(i));
      CHECK(cudaEventCreate(&starts[i]));
      CHECK(cudaEventCreate(&stops[i]));
    }

    // Use a CPU barrier to get all streams launching close together.
    std::barrier round_barrier(ngpu);
    std::vector<double> per_gpu_bw(ngpu);

    std::vector<std::thread> round_threads;
    for (int i = 0; i < ngpu; i++) {
      round_threads.emplace_back([&, i, offset]() {
        int peer = (i + offset) % ngpu;
        CHECK(cudaSetDevice(i));

        round_barrier.arrive_and_wait();

        CHECK(cudaEventRecord(starts[i], stag_streams[i]));
        for (int it = 0; it < ITERS; it++)
          CHECK(cudaMemcpyPeerAsync(stag_dst[i], i, stag_bufs[peer], peer,
                                    TRANSFER_SIZE, stag_streams[i]));
        CHECK(cudaEventRecord(stops[i], stag_streams[i]));
        CHECK(cudaStreamSynchronize(stag_streams[i]));

        float ms;
        CHECK(cudaEventElapsedTime(&ms, starts[i], stops[i]));
        per_gpu_bw[i] = (double)TRANSFER_SIZE * ITERS / (ms / 1000.0) / 1e9;
      });
    }
    for (auto& t : round_threads) t.join();

    double total = 0;
    for (int i = 0; i < ngpu; i++) total += per_gpu_bw[i];
    printf("+%d  ", offset);
    for (int i = 0; i < ngpu; i++) {
      int peer = (i + offset) % ngpu;
      printf("%d<-%d ", i, peer);
    }
    printf(" %6.2f avg  %7.2f total\n", total / ngpu, total);

    for (int i = 0; i < ngpu; i++) {
      CHECK(cudaSetDevice(i));
      CHECK(cudaEventDestroy(starts[i]));
      CHECK(cudaEventDestroy(stops[i]));
    }
  }

  for (int i = 0; i < ngpu; i++) {
    CHECK(cudaSetDevice(i));
    CHECK(cudaFree(stag_bufs[i]));
    CHECK(cudaFree(stag_dst[i]));
    CHECK(cudaStreamDestroy(stag_streams[i]));
  }

  // ---- Test 2b: Single reader, all peers concurrent ----
  // One GPU reads from all N-1 peers at once (N-1 streams), no other
  // GPUs active. Shows the max inbound bandwidth per GPU without
  // cross-traffic from other readers.
  printf("\n=== Single reader, all %d peers concurrent (GB/s) ===\n\n", ngpu - 1);

  std::vector<void*> sr_bufs(ngpu);
  for (int i = 0; i < ngpu; i++) {
    CHECK(cudaSetDevice(i));
    CHECK(cudaMalloc(&sr_bufs[i], TRANSFER_SIZE));
    CHECK(cudaMemset(sr_bufs[i], i, TRANSFER_SIZE));
  }

  auto sr_wall_t0 = std::chrono::high_resolution_clock::now();
  double sr_total_bytes = 0;
  std::vector<double> sr_bw(ngpu);

  for (int reader = 0; reader < ngpu; reader++) {
    CHECK(cudaSetDevice(reader));
    std::vector<void*> dst(ngpu - 1);
    std::vector<cudaStream_t> streams(ngpu - 1);
    std::vector<int> peers(ngpu - 1);
    for (int r = 0; r < ngpu - 1; r++) {
      peers[r] = (reader + r + 1) % ngpu;
      CHECK(cudaMalloc(&dst[r], TRANSFER_SIZE));
      CHECK(cudaStreamCreate(&streams[r]));
    }

    // Warmup.
    for (int r = 0; r < ngpu - 1; r++) {
      for (int w = 0; w < WARMUP; w++)
        CHECK(cudaMemcpyPeerAsync(dst[r], reader, sr_bufs[peers[r]], peers[r],
                                  TRANSFER_SIZE, streams[r]));
    }
    for (int r = 0; r < ngpu - 1; r++)
      CHECK(cudaStreamSynchronize(streams[r]));

    // Timed.
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < ITERS; it++) {
      for (int r = 0; r < ngpu - 1; r++)
        CHECK(cudaMemcpyPeerAsync(dst[r], reader, sr_bufs[peers[r]], peers[r],
                                  TRANSFER_SIZE, streams[r]));
    }
    for (int r = 0; r < ngpu - 1; r++)
      CHECK(cudaStreamSynchronize(streams[r]));
    auto t1 = std::chrono::high_resolution_clock::now();

    double sec = std::chrono::duration<double>(t1 - t0).count();
    double bytes = (double)(ngpu - 1) * TRANSFER_SIZE * ITERS;
    double gbps = bytes / sec / 1e9;

    sr_bw[reader] = gbps;
    printf("GPU %d reads from all peers: %.2f GB/s\n", reader, gbps);
    sr_total_bytes += bytes;

    for (int r = 0; r < ngpu - 1; r++) {
      CHECK(cudaFree(dst[r]));
      CHECK(cudaStreamDestroy(streams[r]));
    }
  }

  auto sr_wall_t1 = std::chrono::high_resolution_clock::now();
  double sr_wall_sec = std::chrono::duration<double>(sr_wall_t1 - sr_wall_t0).count();
  printf("\nEffective system bandwidth (sequential): %.2f GB/s\n",
         sr_total_bytes / sr_wall_sec / 1e9);

  for (int i = 0; i < ngpu; i++) {
    CHECK(cudaSetDevice(i));
    CHECK(cudaFree(sr_bufs[i]));
  }

  // ---- Test 3: All GPUs read from all peers simultaneously ----
  // Same as test 2b but all 8 GPUs go at once. One thread per GPU,
  // N-1 streams each, all timed together.
  printf("\n=== All GPUs read all peers simultaneously (GB/s) ===\n");
  printf("Each GPU has %d streams, %d total concurrent transfers.\n\n",
         ngpu - 1, ngpu * (ngpu - 1));

  static constexpr int FC_ITERS = 100;
  static constexpr int FC_WARMUP = 20;

  std::vector<void*> fc_bufs(ngpu);
  std::vector<std::vector<void*>> fc_dst(ngpu, std::vector<void*>(ngpu - 1));
  std::vector<std::vector<cudaStream_t>> fc_streams(ngpu, std::vector<cudaStream_t>(ngpu - 1));

  for (int i = 0; i < ngpu; i++) {
    CHECK(cudaSetDevice(i));
    CHECK(cudaMalloc(&fc_bufs[i], TRANSFER_SIZE));
    CHECK(cudaMemset(fc_bufs[i], i, TRANSFER_SIZE));
    for (int r = 0; r < ngpu - 1; r++) {
      CHECK(cudaMalloc(&fc_dst[i][r], TRANSFER_SIZE));
      CHECK(cudaStreamCreate(&fc_streams[i][r]));
    }
  }

  // Warmup.
  for (int i = 0; i < ngpu; i++) {
    CHECK(cudaSetDevice(i));
    for (int r = 0; r < ngpu - 1; r++) {
      int peer = (i + r + 1) % ngpu;
      for (int w = 0; w < FC_WARMUP; w++)
        CHECK(cudaMemcpyPeerAsync(fc_dst[i][r], i, fc_bufs[peer], peer,
                                  TRANSFER_SIZE, fc_streams[i][r]));
    }
  }
  for (int i = 0; i < ngpu; i++) {
    CHECK(cudaSetDevice(i));
    for (int r = 0; r < ngpu - 1; r++)
      CHECK(cudaStreamSynchronize(fc_streams[i][r]));
  }

  std::barrier fc_barrier(ngpu);
  std::vector<double> fc_bw(ngpu);
  std::vector<std::thread> fc_threads;

  for (int i = 0; i < ngpu; i++) {
    fc_threads.emplace_back([&, i]() {
      CHECK(cudaSetDevice(i));

      fc_barrier.arrive_and_wait();

      auto t0 = std::chrono::high_resolution_clock::now();
      for (int it = 0; it < FC_ITERS; it++) {
        for (int r = 0; r < ngpu - 1; r++) {
          int peer = (i + r + 1) % ngpu;
          CHECK(cudaMemcpyPeerAsync(fc_dst[i][r], i, fc_bufs[peer], peer,
                                    TRANSFER_SIZE, fc_streams[i][r]));
        }
      }
      for (int r = 0; r < ngpu - 1; r++)
        CHECK(cudaStreamSynchronize(fc_streams[i][r]));
      auto t1 = std::chrono::high_resolution_clock::now();

      double sec = std::chrono::duration<double>(t1 - t0).count();
      double bytes = (double)(ngpu - 1) * TRANSFER_SIZE * FC_ITERS;
      fc_bw[i] = bytes / sec / 1e9;
    });
  }
  for (auto& t : fc_threads) t.join();

  double total_fc = 0;
  double ideal_total = 0;
  for (int i = 0; i < ngpu; i++) {
    printf("GPU %d: %.2f GB/s\n", i, fc_bw[i]);
    total_fc += fc_bw[i];
    ideal_total += sr_bw[i];
  }
  printf("\nTotal system bandwidth: %.2f GB/s\n", total_fc);
  double avg_per_gpu = ideal_total / ngpu;
  static constexpr double PCIE_X16_THEORETICAL = 63.0;  // PCIe 5.0 x16 = 64 GT/s ~ 63 GB/s

  printf("\n");
  printf("===========================================================\n");
  printf("  PCIe LINK SCORE:           %.2f\n", avg_per_gpu / PCIE_X16_THEORETICAL);
  printf("  (%.2f GB/s avg  /  %.1f GB/s PCIe 5.0 x16 theoretical)\n",
         avg_per_gpu, PCIE_X16_THEORETICAL);
  printf("\n");
  printf("  DENSE INTERCONNECT SCORE:  %.2f\n", total_fc / ideal_total);
  printf("  (%.2f GB/s measured  /  %.2f GB/s ideal)\n", total_fc, ideal_total);
  printf("\n");
  printf("  1.00 = perfect, 0.00 = none\n");
  printf("===========================================================\n");

  for (int i = 0; i < ngpu; i++) {
    CHECK(cudaSetDevice(i));
    CHECK(cudaFree(fc_bufs[i]));
    for (int r = 0; r < ngpu - 1; r++) {
      CHECK(cudaFree(fc_dst[i][r]));
      CHECK(cudaStreamDestroy(fc_streams[i][r]));
    }
  }

  return 0;
}

// ---- Latency mode ----

// Measure latency of a single remote read (8KB) from reader GPU reading src_buf
// on another GPU. Returns average latency in microseconds.
static double measure_latency(int reader, void* src_buf, void* local_buf,
                              cudaStream_t stream) {
  // Warmup.
  for (int i = 0; i < LATENCY_WARMUP; i++)
    latency_read_kernel<<<1, 1, 0, stream>>>((const int*)src_buf, (int*)local_buf, LATENCY_WORDS);
  CHECK(cudaStreamSynchronize(stream));

  cudaEvent_t start, stop;
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&stop));
  CHECK(cudaEventRecord(start, stream));
  for (int i = 0; i < LATENCY_ITERS; i++)
    latency_read_kernel<<<1, 1, 0, stream>>>((const int*)src_buf, (int*)local_buf, LATENCY_WORDS);
  CHECK(cudaEventRecord(stop, stream));
  CHECK(cudaStreamSynchronize(stream));

  float ms;
  CHECK(cudaEventElapsedTime(&ms, start, stop));
  CHECK(cudaEventDestroy(start));
  CHECK(cudaEventDestroy(stop));
  return (double)ms * 1000.0 / LATENCY_ITERS;  // microseconds
}

static void run_latency_tests(int ngpu) {
  printf("=== LATENCY MODE (%zu-byte remote reads, %d iterations) ===\n\n", LATENCY_SIZE, LATENCY_ITERS);

  // Allocate one source buffer and one local scratch per GPU.
  std::vector<void*> bufs(ngpu), local(ngpu);
  std::vector<cudaStream_t> streams(ngpu);
  for (int i = 0; i < ngpu; i++) {
    CHECK(cudaSetDevice(i));
    CHECK(cudaMalloc(&bufs[i], LATENCY_SIZE));
    CHECK(cudaMemset(bufs[i], i, LATENCY_SIZE));
    CHECK(cudaMalloc(&local[i], sizeof(int)));
    CHECK(cudaStreamCreate(&streams[i]));
  }

  // ---- Test 1: Sequential NxN latency ----
  printf("=== Sequential P2P latency (us) ===\n\n");

  std::vector<std::vector<double>> seq_lat(ngpu, std::vector<double>(ngpu, 0.0));

  printf("%6s", "Src->");
  for (int s = 0; s < ngpu; s++) printf("  GPU%-4d", s);
  printf("\n");

  for (int r = 0; r < ngpu; r++) {
    printf("GPU %d:", r);
    CHECK(cudaSetDevice(r));
    for (int s = 0; s < ngpu; s++) {
      if (r == s) {
        printf("     -   ");
        continue;
      }
      double us = measure_latency(r, bufs[s], local[r], streams[r]);
      seq_lat[r][s] = us;
      printf("  %6.2f  ", us);
    }
    printf("\n");
  }

  double seq_avg = 0;
  int seq_count = 0;
  for (int r = 0; r < ngpu; r++)
    for (int s = 0; s < ngpu; s++)
      if (r != s) { seq_avg += seq_lat[r][s]; seq_count++; }
  seq_avg /= seq_count;
  printf("\nAverage 1:1 latency: %.2f us\n", seq_avg);

  // ---- Test 2: Topology probe latency ----
  printf("\n=== Topology probe: staggered reads by peer distance (us) ===\n");
  printf("%d concurrent kernel launches per round.\n\n", ngpu);

  for (int r = 0; r < ngpu - 1; r++) {
    int offset = r + 1;

    // Warmup.
    for (int i = 0; i < ngpu; i++) {
      int peer = (i + offset) % ngpu;
      CHECK(cudaSetDevice(i));
      for (int w = 0; w < LATENCY_WARMUP; w++)
        latency_read_kernel<<<1, 1, 0, streams[i]>>>((const int*)bufs[peer], (int*)local[i], LATENCY_WORDS);
    }
    for (int i = 0; i < ngpu; i++) {
      CHECK(cudaSetDevice(i));
      CHECK(cudaStreamSynchronize(streams[i]));
    }

    // Timed.
    std::vector<cudaEvent_t> starts(ngpu), stops(ngpu);
    for (int i = 0; i < ngpu; i++) {
      CHECK(cudaSetDevice(i));
      CHECK(cudaEventCreate(&starts[i]));
      CHECK(cudaEventCreate(&stops[i]));
    }

    std::barrier round_barrier(ngpu);
    std::vector<double> per_gpu_lat(ngpu);
    std::vector<std::thread> round_threads;

    for (int i = 0; i < ngpu; i++) {
      round_threads.emplace_back([&, i, offset]() {
        int peer = (i + offset) % ngpu;
        CHECK(cudaSetDevice(i));

        round_barrier.arrive_and_wait();

        CHECK(cudaEventRecord(starts[i], streams[i]));
        for (int it = 0; it < LATENCY_ITERS; it++)
          latency_read_kernel<<<1, 1, 0, streams[i]>>>((const int*)bufs[peer], (int*)local[i], LATENCY_WORDS);
        CHECK(cudaEventRecord(stops[i], streams[i]));
        CHECK(cudaStreamSynchronize(streams[i]));

        float ms;
        CHECK(cudaEventElapsedTime(&ms, starts[i], stops[i]));
        per_gpu_lat[i] = (double)ms * 1000.0 / LATENCY_ITERS;
      });
    }
    for (auto& t : round_threads) t.join();

    double avg = 0;
    for (int i = 0; i < ngpu; i++) avg += per_gpu_lat[i];
    avg /= ngpu;
    printf("+%d  ", offset);
    for (int i = 0; i < ngpu; i++) {
      int peer = (i + offset) % ngpu;
      printf("%d<-%d ", i, peer);
    }
    printf(" %6.2f us avg\n", avg);

    for (int i = 0; i < ngpu; i++) {
      CHECK(cudaSetDevice(i));
      CHECK(cudaEventDestroy(starts[i]));
      CHECK(cudaEventDestroy(stops[i]));
    }
  }

  // ---- Test 2b: Single reader, all peers concurrent ----
  printf("\n=== Single reader, all %d peers concurrent (us) ===\n\n", ngpu - 1);

  std::vector<double> sr_lat(ngpu);

  for (int reader = 0; reader < ngpu; reader++) {
    CHECK(cudaSetDevice(reader));
    std::vector<cudaStream_t> peer_streams(ngpu - 1);
    std::vector<void*> peer_local(ngpu - 1);
    for (int r = 0; r < ngpu - 1; r++) {
      CHECK(cudaStreamCreate(&peer_streams[r]));
      CHECK(cudaMalloc(&peer_local[r], sizeof(int)));
    }

    // Warmup.
    for (int r = 0; r < ngpu - 1; r++) {
      int peer = (reader + r + 1) % ngpu;
      for (int w = 0; w < LATENCY_WARMUP; w++)
        latency_read_kernel<<<1, 1, 0, peer_streams[r]>>>((const int*)bufs[peer], (int*)peer_local[r], LATENCY_WORDS);
    }
    for (int r = 0; r < ngpu - 1; r++)
      CHECK(cudaStreamSynchronize(peer_streams[r]));

    // Timed: all streams launch concurrently.
    std::vector<cudaEvent_t> starts(ngpu - 1), stops(ngpu - 1);
    for (int r = 0; r < ngpu - 1; r++) {
      CHECK(cudaEventCreate(&starts[r]));
      CHECK(cudaEventCreate(&stops[r]));
    }
    for (int r = 0; r < ngpu - 1; r++) {
      int peer = (reader + r + 1) % ngpu;
      CHECK(cudaEventRecord(starts[r], peer_streams[r]));
      for (int it = 0; it < LATENCY_ITERS; it++)
        latency_read_kernel<<<1, 1, 0, peer_streams[r]>>>((const int*)bufs[peer], (int*)peer_local[r], LATENCY_WORDS);
      CHECK(cudaEventRecord(stops[r], peer_streams[r]));
    }
    for (int r = 0; r < ngpu - 1; r++)
      CHECK(cudaStreamSynchronize(peer_streams[r]));

    double max_lat = 0;
    for (int r = 0; r < ngpu - 1; r++) {
      float ms;
      CHECK(cudaEventElapsedTime(&ms, starts[r], stops[r]));
      double us = (double)ms * 1000.0 / LATENCY_ITERS;
      max_lat = std::max(max_lat, us);
    }
    sr_lat[reader] = max_lat;
    printf("GPU %d: %.2f us (max across %d peers)\n", reader, max_lat, ngpu - 1);

    for (int r = 0; r < ngpu - 1; r++) {
      CHECK(cudaEventDestroy(starts[r]));
      CHECK(cudaEventDestroy(stops[r]));
      CHECK(cudaFree(peer_local[r]));
      CHECK(cudaStreamDestroy(peer_streams[r]));
    }
  }

  // ---- Test 3: All GPUs read all peers simultaneously ----
  printf("\n=== All GPUs read all peers simultaneously (us) ===\n");
  printf("Each GPU has %d streams, %d total concurrent kernels.\n\n",
         ngpu - 1, ngpu * (ngpu - 1));

  // Allocate per-GPU, per-peer streams and scratch.
  std::vector<std::vector<cudaStream_t>> ac_streams(ngpu, std::vector<cudaStream_t>(ngpu - 1));
  std::vector<std::vector<void*>> ac_local(ngpu, std::vector<void*>(ngpu - 1));
  for (int i = 0; i < ngpu; i++) {
    CHECK(cudaSetDevice(i));
    for (int r = 0; r < ngpu - 1; r++) {
      CHECK(cudaStreamCreate(&ac_streams[i][r]));
      CHECK(cudaMalloc(&ac_local[i][r], sizeof(int)));
    }
  }

  // Warmup.
  for (int i = 0; i < ngpu; i++) {
    CHECK(cudaSetDevice(i));
    for (int r = 0; r < ngpu - 1; r++) {
      int peer = (i + r + 1) % ngpu;
      for (int w = 0; w < LATENCY_WARMUP; w++)
        latency_read_kernel<<<1, 1, 0, ac_streams[i][r]>>>((const int*)bufs[peer], (int*)ac_local[i][r], LATENCY_WORDS);
    }
  }
  for (int i = 0; i < ngpu; i++) {
    CHECK(cudaSetDevice(i));
    for (int r = 0; r < ngpu - 1; r++)
      CHECK(cudaStreamSynchronize(ac_streams[i][r]));
  }

  // Timed.
  std::barrier ac_barrier(ngpu);
  std::vector<double> ac_lat(ngpu);
  std::vector<std::thread> ac_threads;

  for (int i = 0; i < ngpu; i++) {
    ac_threads.emplace_back([&, i]() {
      CHECK(cudaSetDevice(i));

      ac_barrier.arrive_and_wait();

      // Launch all peer kernels, measure per-stream.
      std::vector<cudaEvent_t> starts(ngpu - 1), stops(ngpu - 1);
      for (int r = 0; r < ngpu - 1; r++) {
        CHECK(cudaEventCreate(&starts[r]));
        CHECK(cudaEventCreate(&stops[r]));
      }
      for (int r = 0; r < ngpu - 1; r++) {
        int peer = (i + r + 1) % ngpu;
        CHECK(cudaEventRecord(starts[r], ac_streams[i][r]));
        for (int it = 0; it < LATENCY_ITERS; it++)
          latency_read_kernel<<<1, 1, 0, ac_streams[i][r]>>>((const int*)bufs[peer], (int*)ac_local[i][r], LATENCY_WORDS);
        CHECK(cudaEventRecord(stops[r], ac_streams[i][r]));
      }
      for (int r = 0; r < ngpu - 1; r++)
        CHECK(cudaStreamSynchronize(ac_streams[i][r]));

      double max_lat = 0;
      for (int r = 0; r < ngpu - 1; r++) {
        float ms;
        CHECK(cudaEventElapsedTime(&ms, starts[r], stops[r]));
        double us = (double)ms * 1000.0 / LATENCY_ITERS;
        max_lat = std::max(max_lat, us);
        CHECK(cudaEventDestroy(starts[r]));
        CHECK(cudaEventDestroy(stops[r]));
      }
      ac_lat[i] = max_lat;
    });
  }
  for (auto& t : ac_threads) t.join();

  double ac_avg = 0;
  for (int i = 0; i < ngpu; i++) {
    printf("GPU %d: %.2f us (max across %d peers)\n", i, ac_lat[i], ngpu - 1);
    ac_avg += ac_lat[i];
  }
  ac_avg /= ngpu;

  // Find min and max from sequential NxN.
  double seq_min = 1e9;
  for (int i = 0; i < ngpu; i++)
    for (int j = 0; j < ngpu; j++)
      if (i != j) seq_min = std::min(seq_min, seq_lat[i][j]);

  // Find max from all-concurrent.
  double ac_max = 0;
  for (int i = 0; i < ngpu; i++)
    ac_max = std::max(ac_max, ac_lat[i]);

  printf("\n");
  printf("===========================================================\n");
  printf("  P2P LATENCY (1:1):     %6.2f us  (best pair)\n", seq_min);
  printf("  LOADED LATENCY (avg):  %6.2f us  (all GPUs concurrent)\n", ac_avg);
  printf("  LOADED LATENCY (max):  %6.2f us  (worst GPU concurrent)\n", ac_max);
  printf("===========================================================\n");

  // Cleanup.
  for (int i = 0; i < ngpu; i++) {
    CHECK(cudaSetDevice(i));
    for (int r = 0; r < ngpu - 1; r++) {
      CHECK(cudaFree(ac_local[i][r]));
      CHECK(cudaStreamDestroy(ac_streams[i][r]));
    }
  }
  for (int i = 0; i < ngpu; i++) {
    CHECK(cudaSetDevice(i));
    CHECK(cudaFree(bufs[i]));
    CHECK(cudaFree(local[i]));
    CHECK(cudaStreamDestroy(streams[i]));
  }
}
