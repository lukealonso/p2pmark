#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <nccl.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <thread>
#include <barrier>
#include <chrono>

#define CHECK_NCCL(cmd) do {                                                \
  ncclResult_t r = cmd;                                                     \
  if (r != ncclSuccess) {                                                   \
    fprintf(stderr, "NCCL error %s:%d: %s\n", __FILE__, __LINE__,          \
            ncclGetErrorString(r));                                          \
    exit(1);                                                                \
  }                                                                         \
} while (0)

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
static constexpr int NVIDIA_NUM_ELEMS = 40000000;  // NVIDIA's default: 40M ints
static constexpr size_t TRANSFER_SIZE = (size_t)NVIDIA_NUM_ELEMS * sizeof(int);  // 160,000,000 bytes
static constexpr int WARMUP = 5;
static constexpr int ITERS = 5;

// Latency mode constants.
static constexpr size_t LATENCY_SIZE = 128;  // one cacheline
static constexpr int LATENCY_WORDS = LATENCY_SIZE / sizeof(int);
static constexpr int LATENCY_ITERS = 10000;

// ---- PCIe allreduce kernel (from pcie_allreduce.cu, torch deps stripped) ----

namespace pcie_ar {

constexpr int kMaxBlocks = 36;
using FlagType = uint32_t;
constexpr int kFlagStride = 32;

struct Signal {
  alignas(128) FlagType self_counter[kMaxBlocks][8];
  alignas(128) FlagType peer_counter[2][kMaxBlocks][16 * kFlagStride];
};

struct __align__(16) RankData {
  const void* __restrict__ ptrs[8];
};

struct __align__(16) RankSignals {
  Signal* signals[8];
};

template <typename T, int sz>
struct __align__(alignof(T) * sz) array_t {
  T data[sz];
  using type = T;
  static constexpr int size = sz;
};

template <typename T>
struct packed_t {
  using P = array_t<T, 16 / sizeof(T)>;
  using A = array_t<float, 16 / sizeof(T)>;
};

#define DINLINE __device__ __forceinline__

DINLINE float upcast_s(half val) { return __half2float(val); }
template <typename T> DINLINE T downcast_s(float val);
template <> DINLINE half downcast_s(float val) { return __float2half(val); }
DINLINE half& assign_add(half& a, half b) { a = __hadd(a, b); return a; }
DINLINE float& assign_add(float& a, float b) { return a += b; }

#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
DINLINE float upcast_s(nv_bfloat16 val) { return __bfloat162float(val); }
template <> DINLINE nv_bfloat16 downcast_s(float val) { return __float2bfloat16(val); }
DINLINE nv_bfloat16& assign_add(nv_bfloat16& a, nv_bfloat16 b) { a = __hadd(a, b); return a; }
#endif

template <typename T, int N>
DINLINE array_t<T, N>& packed_assign_add(array_t<T, N>& a, array_t<T, N> b) {
#pragma unroll
  for (int i = 0; i < N; i++) assign_add(a.data[i], b.data[i]);
  return a;
}

template <typename T, int N>
DINLINE array_t<float, N> upcast(array_t<T, N> val) {
  if constexpr (std::is_same<T, float>::value) {
    return val;
  } else {
    array_t<float, N> out;
#pragma unroll
    for (int i = 0; i < N; i++) out.data[i] = upcast_s(val.data[i]);
    return out;
  }
}

template <typename O>
DINLINE O downcast(array_t<float, O::size> val) {
  if constexpr (std::is_same<typename O::type, float>::value) {
    return val;
  } else {
    O out;
#pragma unroll
    for (int i = 0; i < O::size; i++) out.data[i] = downcast_s<typename O::type>(val.data[i]);
    return out;
  }
}

static DINLINE void st_flag_relaxed(FlagType* flag_addr, FlagType flag) {
  asm volatile("st.relaxed.sys.global.u32 [%1], %0;" ::"r"(flag), "l"(flag_addr));
}

static DINLINE FlagType ld_flag_relaxed(FlagType* flag_addr) {
  FlagType flag;
  asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(flag) : "l"(flag_addr));
  return flag;
}

template <int ngpus, bool is_start>
DINLINE void multi_gpu_barrier(const RankSignals& sg, Signal* self_sg, int rank) {
  if constexpr (!is_start) __syncthreads();
  if (threadIdx.x < ngpus) {
    __threadfence_system();
    auto val = self_sg->self_counter[blockIdx.x][threadIdx.x] += 1;
    auto peer_counter_ptr = &sg.signals[threadIdx.x]->peer_counter[val % 2][blockIdx.x][rank * kFlagStride];
    auto self_counter_ptr = &self_sg->peer_counter[val % 2][blockIdx.x][threadIdx.x * kFlagStride];
    st_flag_relaxed(peer_counter_ptr, val);
    while (ld_flag_relaxed(self_counter_ptr) != val);
  }
  __syncthreads();
}

template <typename P, int ngpus, typename A>
DINLINE P packed_reduce(const P* ptrs[], int idx) {
  A tmp = upcast(ptrs[0][idx]);
#pragma unroll
  for (int i = 1; i < ngpus; i++) packed_assign_add(tmp, upcast(ptrs[i][idx]));
  return downcast<P>(tmp);
}

template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1) pcie_allreduce_kernel(
    RankData* _dp, RankSignals sg, Signal* self_sg, T* __restrict__ result, int rank, int size) {
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  auto dp = *_dp;
  const P* rotated[ngpus];
#pragma unroll
  for (int i = 0; i < ngpus; i++) {
    rotated[i] = (const P*)dp.ptrs[(rank + i) % ngpus];
  }
  multi_gpu_barrier<ngpus, true>(sg, self_sg, rank);
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += gridDim.x * blockDim.x) {
    ((P*)result)[idx] = packed_reduce<P, ngpus, A>(rotated, idx);
  }
}

template <typename T>
static void launch_kernel(int ngpus, int blocks, int threads, cudaStream_t stream,
                          RankData* rd, RankSignals rs, Signal* self_sg,
                          T* output, int rank, int packed_size) {
#define KL(ng) pcie_allreduce_kernel<T, ng><<<blocks, threads, 0, stream>>>(rd, rs, self_sg, output, rank, packed_size)
  switch (ngpus) {
    case 2: KL(2); break;
    case 3: KL(3); break;
    case 4: KL(4); break;
    case 5: KL(5); break;
    case 6: KL(6); break;
    case 7: KL(7); break;
    case 8: KL(8); break;
  }
#undef KL
}

#undef DINLINE

}  // namespace pcie_ar

// NVIDIA's delay kernel — blocks stream until flag is set, ensuring all
// subsequent copies are queued before the GPU starts processing them.
// This eliminates CPU-side API overhead gaps in the DMA pipeline.
__global__ void delay_kernel(volatile int *flag, unsigned long long timeout_clocks = 200000000) {
  long long int start_clock = clock64();
  while (!*flag) {
    if (clock64() - start_clock > timeout_clocks) break;
  }
}

// Cache-volatile load — bypasses L2 so every read goes over PCIe/NVLink.
__device__ __forceinline__ int load_cv(const int* addr) {
  int val;
  asm volatile("ld.global.cv.s32 %0, [%1];" : "=r"(val) : "l"(addr));
  return val;
}

// Single-thread kernel that reads `words` ints from a remote GPU pointer
// `iters` times using cache-volatile loads, returning elapsed clock cycles
// via `out_cycles`. P2P access must be enabled so `remote` (allocated on
// another GPU) is directly accessible from the launching GPU.
__global__ void latency_read_kernel(const int* __restrict__ remote, int* local,
                                    int words, int iters, long long* out_cycles) {
  // Warmup (not timed).
  int sum = 0;
  for (int i = 0; i < words; i++)
    sum += load_cv(remote + i);

  long long t0 = clock64();
  for (int it = 0; it < iters; it++) {
    for (int i = 0; i < words; i++)
      sum += load_cv(remote + i);
  }
  long long t1 = clock64();

  local[0] = sum;
  *out_cycles = t1 - t0;
}

// Measure unidirectional D2D write bandwidth from src GPU to dst GPU.
// Uses cudaEvent timing to exclude CPU-side API overhead, which is
// significant for cross-socket (xGMI) transfers.
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

  cudaEvent_t e0, e1;
  CHECK(cudaEventCreate(&e0));
  CHECK(cudaEventCreate(&e1));

  CHECK(cudaEventRecord(e0, stream));
  for (int i = 0; i < ITERS; i++)
    CHECK(cudaMemcpyPeerAsync(dst_buf, dst, src_buf, src, bytes, stream));
  CHECK(cudaEventRecord(e1, stream));
  CHECK(cudaStreamSynchronize(stream));

  float ms;
  CHECK(cudaEventElapsedTime(&ms, e0, e1));
  double gbps = (double)bytes * ITERS / (ms / 1000.0) / 1e9;

  CHECK(cudaEventDestroy(e0));
  CHECK(cudaEventDestroy(e1));
  CHECK(cudaSetDevice(src));
  CHECK(cudaFree(src_buf));
  CHECK(cudaSetDevice(dst));
  CHECK(cudaFree(dst_buf));

  return gbps;
}

static void run_latency_tests(int ngpu);
static void run_allreduce_tests(int ngpu);

int main(int argc, char** argv) {
  bool latency_mode = false;
  bool allreduce_mode = false;
  for (int i = 1; i < argc; i++) {
    if (std::string(argv[i]) == "--latency") latency_mode = true;
    if (std::string(argv[i]) == "--allreduce") allreduce_mode = true;
  }

  int ngpu;
  CHECK(cudaGetDeviceCount(&ngpu));
  printf("Found %d GPUs\n\n", ngpu);

  if (ngpu > 8) {
    printf("Clamping to 8 GPUs (P2P hardware limit)\n\n");
    ngpu = 8;
  }

  auto enable_all_peer_access = [&]() {
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
  };

  if (latency_mode) {
    enable_all_peer_access();
    run_latency_tests(ngpu);
    return 0;
  }
  if (allreduce_mode) {
    enable_all_peer_access();
    run_allreduce_tests(ngpu);
    return 0;
  }

  // ---- Test 1: Sequential NxN ----
  // Pre-allocate all buffers (like NVIDIA's test) to avoid allocator
  // address churn that can affect cross-socket xGMI routing.
  printf("=== Sequential P2P bandwidth (GB/s) ===\n");
  printf("Each transfer: %zu MB, %d iterations\n\n", TRANSFER_SIZE / (1024 * 1024), ITERS);

  // Allocate buffers on ALL GPUs BEFORE enabling peer access.
  // This order matches NVIDIA's p2pBandwidthLatencyTest and produces
  // optimal cross-socket xGMI bandwidth.
  std::vector<void*> seq_bufs(ngpu);
  std::vector<cudaStream_t> seq_streams(ngpu);
  std::vector<cudaEvent_t> seq_starts(ngpu), seq_stops(ngpu);
  for (int i = 0; i < ngpu; i++) {
    CHECK(cudaSetDevice(i));
    CHECK(cudaStreamCreateWithFlags(&seq_streams[i], cudaStreamNonBlocking));
    CHECK(cudaMalloc(&seq_bufs[i], TRANSFER_SIZE));
    CHECK(cudaMemset(seq_bufs[i], 0, TRANSFER_SIZE));
    CHECK(cudaEventCreate(&seq_starts[i]));
    CHECK(cudaEventCreate(&seq_stops[i]));
  }

  // Enable P2P access AFTER allocation (critical for xGMI bandwidth).
  enable_all_peer_access();

  // Host-pinned flag for delay kernel synchronization.
  volatile int *delay_flag;
  CHECK(cudaHostAlloc((void**)&delay_flag, sizeof(*delay_flag), cudaHostAllocPortable));

  std::vector<std::vector<double>> seq_bw(ngpu, std::vector<double>(ngpu, 0.0));

  printf("%6s", "Dst->");
  for (int d = 0; d < ngpu; d++) printf("  GPU%-4d", d);
  printf("\n");

  for (int s = 0; s < ngpu; s++) {
    printf("GPU %d:", s);
    CHECK(cudaSetDevice(s));
    for (int d = 0; d < ngpu; d++) {
      if (s == d) {
        printf("     -   ");
        continue;
      }
      // Timed — delay kernel pre-queues all copies (NVIDIA approach).
      CHECK(cudaStreamSynchronize(seq_streams[s]));
      *delay_flag = 0;
      delay_kernel<<<1, 1, 0, seq_streams[s]>>>(delay_flag);
      CHECK(cudaEventRecord(seq_starts[s], seq_streams[s]));
      for (int it = 0; it < ITERS; it++)
        CHECK(cudaMemcpyPeerAsync(seq_bufs[d], d, seq_bufs[s], s,
                                  TRANSFER_SIZE, seq_streams[s]));
      CHECK(cudaEventRecord(seq_stops[s], seq_streams[s]));
      *delay_flag = 1;
      CHECK(cudaStreamSynchronize(seq_streams[s]));

      float ms;
      CHECK(cudaEventElapsedTime(&ms, seq_starts[s], seq_stops[s]));
      double bw = (double)TRANSFER_SIZE * ITERS / (ms / 1000.0) / 1e9;
      seq_bw[s][d] = bw;
      printf("  %6.2f  ", bw);
    }
    printf("\n");
  }

  for (int i = 0; i < ngpu; i++) {
    CHECK(cudaSetDevice(i));
    CHECK(cudaFree(seq_bufs[i]));
    CHECK(cudaEventDestroy(seq_starts[i]));
    CHECK(cudaEventDestroy(seq_stops[i]));
    CHECK(cudaStreamDestroy(seq_streams[i]));
  }
  CHECK(cudaFreeHost((void*)delay_flag));

  // ---- Test 2: Staggered circular writes (allreduce pattern) ----
  // N-1 rounds. In round r, every GPU i writes to GPU (i+r+1)%N.
  // This is the access pattern used by the PCIe allreduce kernel to
  // spread traffic across the switch fabric.
  printf("\n=== Topology probe: staggered writes by peer distance (GB/s) ===\n");
  printf("%d concurrent transfers per round, each GPU writing to one unique peer.\n", ngpu);
  printf("Reveals PCIe switch topology: nearby peers are fast, cross-switch peers are slow.\n\n");

  std::vector<void*> stag_bufs(ngpu);
  std::vector<void*> stag_dst(ngpu);
  std::vector<cudaStream_t> stag_streams(ngpu);
  volatile int *stag_flag;
  CHECK(cudaHostAlloc((void**)&stag_flag, sizeof(*stag_flag), cudaHostAllocPortable));
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
        CHECK(cudaMemcpyPeerAsync(stag_dst[peer], peer, stag_bufs[i], i,
                                  TRANSFER_SIZE, stag_streams[i]));
    }
    for (int i = 0; i < ngpu; i++) {
      CHECK(cudaSetDevice(i));
      CHECK(cudaStreamSynchronize(stag_streams[i]));
    }

    // Timed: all GPUs launch simultaneously via delay kernel + barrier.
    std::vector<cudaEvent_t> starts(ngpu), stops(ngpu);
    for (int i = 0; i < ngpu; i++) {
      CHECK(cudaSetDevice(i));
      CHECK(cudaEventCreate(&starts[i]));
      CHECK(cudaEventCreate(&stops[i]));
    }

    std::barrier round_barrier(ngpu);
    std::vector<double> per_gpu_bw(ngpu);

    // Pre-queue all copies behind delay kernels on all GPUs.
    *stag_flag = 0;
    for (int i = 0; i < ngpu; i++) {
      int peer = (i + offset) % ngpu;
      CHECK(cudaSetDevice(i));
      delay_kernel<<<1, 1, 0, stag_streams[i]>>>(stag_flag);
      CHECK(cudaEventRecord(starts[i], stag_streams[i]));
      for (int it = 0; it < ITERS; it++)
        CHECK(cudaMemcpyPeerAsync(stag_dst[peer], peer, stag_bufs[i], i,
                                  TRANSFER_SIZE, stag_streams[i]));
      CHECK(cudaEventRecord(stops[i], stag_streams[i]));
    }
    // Release all GPUs simultaneously.
    *stag_flag = 1;

    std::vector<std::thread> round_threads;
    for (int i = 0; i < ngpu; i++) {
      round_threads.emplace_back([&, i]() {
        CHECK(cudaSetDevice(i));
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
      printf("%d->%d ", i, peer);
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
  CHECK(cudaFreeHost((void*)stag_flag));

  // ---- Test 2b: Single GPU writes to all peers concurrent ----
  // One GPU pushes to all N-1 peers at once (N-1 streams), no other
  // GPUs active. Shows the max outbound bandwidth per GPU without
  // cross-traffic.
  printf("\n=== Single GPU writes to all %d peers concurrent (GB/s) ===\n\n", ngpu - 1);

  // Allocate destination buffers on each peer GPU.
  std::vector<std::vector<void*>> sw_dst(ngpu, std::vector<void*>(ngpu - 1));
  std::vector<void*> sw_src(ngpu);
  for (int i = 0; i < ngpu; i++) {
    CHECK(cudaSetDevice(i));
    CHECK(cudaMalloc(&sw_src[i], TRANSFER_SIZE));
    CHECK(cudaMemset(sw_src[i], i, TRANSFER_SIZE));
    for (int r = 0; r < ngpu - 1; r++) {
      int peer = (i + r + 1) % ngpu;
      CHECK(cudaSetDevice(peer));
      CHECK(cudaMalloc(&sw_dst[i][r], TRANSFER_SIZE));
    }
  }

  std::vector<double> sw_bw(ngpu);
  volatile int *sw_flag;
  CHECK(cudaHostAlloc((void**)&sw_flag, sizeof(*sw_flag), cudaHostAllocPortable));

  for (int writer = 0; writer < ngpu; writer++) {
    CHECK(cudaSetDevice(writer));
    std::vector<cudaStream_t> streams(ngpu - 1);
    for (int r = 0; r < ngpu - 1; r++)
      CHECK(cudaStreamCreate(&streams[r]));

    // Warmup.
    for (int r = 0; r < ngpu - 1; r++) {
      int peer = (writer + r + 1) % ngpu;
      for (int w = 0; w < WARMUP; w++)
        CHECK(cudaMemcpyPeerAsync(sw_dst[writer][r], peer, sw_src[writer], writer,
                                  TRANSFER_SIZE, streams[r]));
    }
    for (int r = 0; r < ngpu - 1; r++)
      CHECK(cudaStreamSynchronize(streams[r]));

    // Timed — delay kernel pre-queues all copies across all streams.
    cudaEvent_t ev_start, ev_stop;
    CHECK(cudaEventCreate(&ev_start));
    CHECK(cudaEventCreate(&ev_stop));

    *sw_flag = 0;
    delay_kernel<<<1, 1, 0, streams[0]>>>(sw_flag);
    CHECK(cudaEventRecord(ev_start, streams[0]));
    for (int r = 1; r < ngpu - 1; r++)
      CHECK(cudaStreamWaitEvent(streams[r], ev_start, 0));

    for (int it = 0; it < ITERS; it++) {
      for (int r = 0; r < ngpu - 1; r++) {
        int peer = (writer + r + 1) % ngpu;
        CHECK(cudaMemcpyPeerAsync(sw_dst[writer][r], peer, sw_src[writer], writer,
                                  TRANSFER_SIZE, streams[r]));
      }
    }

    for (int r = 1; r < ngpu - 1; r++) {
      cudaEvent_t done;
      CHECK(cudaEventCreate(&done));
      CHECK(cudaEventRecord(done, streams[r]));
      CHECK(cudaStreamWaitEvent(streams[0], done, 0));
      CHECK(cudaEventDestroy(done));
    }
    CHECK(cudaEventRecord(ev_stop, streams[0]));
    *sw_flag = 1;
    CHECK(cudaStreamSynchronize(streams[0]));

    float ms;
    CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
    double bytes = (double)(ngpu - 1) * TRANSFER_SIZE * ITERS;
    sw_bw[writer] = bytes / (ms / 1000.0) / 1e9;
    printf("GPU %d writes to all peers: %.2f GB/s\n", writer, sw_bw[writer]);

    CHECK(cudaEventDestroy(ev_start));
    CHECK(cudaEventDestroy(ev_stop));
    for (int r = 0; r < ngpu - 1; r++)
      CHECK(cudaStreamDestroy(streams[r]));
  }

  // ---- Test 2c: Ring bandwidth (allreduce pattern) ----
  // Each GPU writes only to GPU (i+1)%N — exactly N concurrent flows.
  // This matches real collective communication patterns (e.g., allreduce ring).
  printf("\n=== Ring bandwidth: each GPU writes to next neighbor (GB/s) ===\n");
  printf("%d concurrent transfers (one per GPU), matching allreduce ring pattern.\n\n", ngpu);

  static constexpr int RING_ITERS = 100;
  static constexpr int RING_WARMUP = 20;

  // Allocate ring buffers: each GPU has one src and one dst.
  std::vector<void*> ring_src(ngpu), ring_dst(ngpu);
  std::vector<cudaStream_t> ring_streams(ngpu);
  for (int i = 0; i < ngpu; i++) {
    CHECK(cudaSetDevice(i));
    CHECK(cudaMalloc(&ring_src[i], TRANSFER_SIZE));
    CHECK(cudaMemset(ring_src[i], i, TRANSFER_SIZE));
    CHECK(cudaMalloc(&ring_dst[i], TRANSFER_SIZE));
    CHECK(cudaStreamCreate(&ring_streams[i]));
  }

  // Warmup.
  for (int i = 0; i < ngpu; i++) {
    int peer = (i + 1) % ngpu;
    CHECK(cudaSetDevice(i));
    for (int w = 0; w < RING_WARMUP; w++)
      CHECK(cudaMemcpyPeerAsync(ring_dst[peer], peer, ring_src[i], i,
                                TRANSFER_SIZE, ring_streams[i]));
  }
  for (int i = 0; i < ngpu; i++) {
    CHECK(cudaSetDevice(i));
    CHECK(cudaStreamSynchronize(ring_streams[i]));
  }

  volatile int *ring_flag;
  CHECK(cudaHostAlloc((void**)&ring_flag, sizeof(*ring_flag), cudaHostAllocPortable));

  std::vector<double> ring_bw(ngpu);
  std::vector<cudaEvent_t> ring_starts(ngpu), ring_stops(ngpu);
  for (int i = 0; i < ngpu; i++) {
    CHECK(cudaSetDevice(i));
    CHECK(cudaEventCreate(&ring_starts[i]));
    CHECK(cudaEventCreate(&ring_stops[i]));
  }

  // Pre-queue all copies behind delay kernels from parallel threads.
  *ring_flag = 0;
  {
    std::barrier ring_queue_barrier(ngpu + 1);
    std::vector<std::thread> ring_queue_threads;
    for (int i = 0; i < ngpu; i++) {
      ring_queue_threads.emplace_back([&, i]() {
        int peer = (i + 1) % ngpu;
        CHECK(cudaSetDevice(i));
        delay_kernel<<<1, 1, 0, ring_streams[i]>>>(ring_flag);
        CHECK(cudaEventRecord(ring_starts[i], ring_streams[i]));
        for (int it = 0; it < RING_ITERS; it++)
          CHECK(cudaMemcpyPeerAsync(ring_dst[peer], peer, ring_src[i], i,
                                    TRANSFER_SIZE, ring_streams[i]));
        CHECK(cudaEventRecord(ring_stops[i], ring_streams[i]));
        ring_queue_barrier.arrive_and_wait();
      });
    }
    ring_queue_barrier.arrive_and_wait();
    *ring_flag = 1;
    for (auto& t : ring_queue_threads) t.join();
  }

  // Collect results.
  std::vector<std::thread> ring_threads;
  for (int i = 0; i < ngpu; i++) {
    ring_threads.emplace_back([&, i]() {
      CHECK(cudaSetDevice(i));
      CHECK(cudaStreamSynchronize(ring_streams[i]));
      float ms;
      CHECK(cudaEventElapsedTime(&ms, ring_starts[i], ring_stops[i]));
      ring_bw[i] = (double)TRANSFER_SIZE * RING_ITERS / (ms / 1000.0) / 1e9;
    });
  }
  for (auto& t : ring_threads) t.join();

  double total_ring = 0;
  for (int i = 0; i < ngpu; i++) {
    int peer = (i + 1) % ngpu;
    printf("GPU %d -> GPU %d: %.2f GB/s\n", i, peer, ring_bw[i]);
    total_ring += ring_bw[i];
  }
  printf("\nTotal ring bandwidth: %.2f GB/s  (%.2f GB/s avg per GPU)\n",
         total_ring, total_ring / ngpu);

  // Cleanup ring resources.
  CHECK(cudaFreeHost((void*)ring_flag));
  for (int i = 0; i < ngpu; i++) {
    CHECK(cudaSetDevice(i));
    CHECK(cudaEventDestroy(ring_starts[i]));
    CHECK(cudaEventDestroy(ring_stops[i]));
    CHECK(cudaFree(ring_src[i]));
    CHECK(cudaFree(ring_dst[i]));
    CHECK(cudaStreamDestroy(ring_streams[i]));
  }

  // ---- Test 3: All GPUs write to all peers simultaneously ----
  // Every GPU pushes to all N-1 peers at once — N*(N-1) concurrent flows.
  // This is a worst-case fabric stress test, NOT a realistic workload.
  // Real collectives (e.g., allreduce ring) use only 2*N flows.
  printf("\n=== All-to-all fabric stress test (GB/s) ===\n");
  printf("Each GPU has %d streams, %d total concurrent transfers [N*(N-1) worst-case].\n\n",
         ngpu - 1, ngpu * (ngpu - 1));

  static constexpr int FC_ITERS = 100;
  static constexpr int FC_WARMUP = 20;

  // Re-use sw_src and sw_dst from test 2b.
  std::vector<std::vector<cudaStream_t>> fc_streams(ngpu, std::vector<cudaStream_t>(ngpu - 1));
  for (int i = 0; i < ngpu; i++) {
    CHECK(cudaSetDevice(i));
    for (int r = 0; r < ngpu - 1; r++)
      CHECK(cudaStreamCreate(&fc_streams[i][r]));
  }

  // Warmup.
  for (int i = 0; i < ngpu; i++) {
    CHECK(cudaSetDevice(i));
    for (int r = 0; r < ngpu - 1; r++) {
      int peer = (i + r + 1) % ngpu;
      for (int w = 0; w < FC_WARMUP; w++)
        CHECK(cudaMemcpyPeerAsync(sw_dst[i][r], peer, sw_src[i], i,
                                  TRANSFER_SIZE, fc_streams[i][r]));
    }
  }
  for (int i = 0; i < ngpu; i++) {
    CHECK(cudaSetDevice(i));
    for (int r = 0; r < ngpu - 1; r++)
      CHECK(cudaStreamSynchronize(fc_streams[i][r]));
  }

  volatile int *fc_flag;
  CHECK(cudaHostAlloc((void**)&fc_flag, sizeof(*fc_flag), cudaHostAllocPortable));

  std::vector<double> fc_bw(ngpu);

  std::vector<cudaEvent_t> fc_starts(ngpu), fc_stops(ngpu);
  for (int i = 0; i < ngpu; i++) {
    CHECK(cudaSetDevice(i));
    CHECK(cudaEventCreate(&fc_starts[i]));
    CHECK(cudaEventCreate(&fc_stops[i]));
  }

  // Pre-queue all copies on all GPUs behind delay kernels from parallel threads.
  *fc_flag = 0;
  {
    std::barrier fc_queue_barrier(ngpu + 1);  // ngpu threads + main thread
    std::vector<std::thread> queue_threads;
    for (int i = 0; i < ngpu; i++) {
      queue_threads.emplace_back([&, i]() {
        CHECK(cudaSetDevice(i));
        delay_kernel<<<1, 1, 0, fc_streams[i][0]>>>(fc_flag);
        CHECK(cudaEventRecord(fc_starts[i], fc_streams[i][0]));
        for (int r = 1; r < ngpu - 1; r++)
          CHECK(cudaStreamWaitEvent(fc_streams[i][r], fc_starts[i], 0));

        for (int it = 0; it < FC_ITERS; it++) {
          for (int r = 0; r < ngpu - 1; r++) {
            int peer = (i + r + 1) % ngpu;
            CHECK(cudaMemcpyPeerAsync(sw_dst[i][r], peer, sw_src[i], i,
                                      TRANSFER_SIZE, fc_streams[i][r]));
          }
        }

        for (int r = 1; r < ngpu - 1; r++) {
          cudaEvent_t done;
          CHECK(cudaEventCreate(&done));
          CHECK(cudaEventRecord(done, fc_streams[i][r]));
          CHECK(cudaStreamWaitEvent(fc_streams[i][0], done, 0));
          CHECK(cudaEventDestroy(done));
        }
        CHECK(cudaEventRecord(fc_stops[i], fc_streams[i][0]));

        // Wait for all threads to finish queuing before releasing.
        fc_queue_barrier.arrive_and_wait();
      });
    }
    // Main thread waits for all queuing to complete, then releases.
    fc_queue_barrier.arrive_and_wait();
    *fc_flag = 1;
    for (auto& t : queue_threads) t.join();
  }

  std::vector<std::thread> fc_threads;
  for (int i = 0; i < ngpu; i++) {
    fc_threads.emplace_back([&, i]() {
      CHECK(cudaSetDevice(i));
      CHECK(cudaStreamSynchronize(fc_streams[i][0]));

      float ms;
      CHECK(cudaEventElapsedTime(&ms, fc_starts[i], fc_stops[i]));
      double bytes = (double)(ngpu - 1) * TRANSFER_SIZE * FC_ITERS;
      fc_bw[i] = bytes / (ms / 1000.0) / 1e9;
    });
  }
  for (auto& t : fc_threads) t.join();

  double total_fc = 0;
  double ideal_total = 0;
  for (int i = 0; i < ngpu; i++) {
    printf("GPU %d: %.2f GB/s\n", i, fc_bw[i]);
    total_fc += fc_bw[i];
    ideal_total += sw_bw[i];
  }
  printf("\nTotal all-to-all bandwidth: %.2f GB/s  (stress test, N*(N-1) flows)\n", total_fc);
  double avg_per_gpu = ideal_total / ngpu;
  static constexpr double PCIE_X16_THEORETICAL = 63.0;  // PCIe 5.0 x16 = 64 GT/s ~ 63 GB/s

  // Ideal ring bandwidth = sum of each link's isolated 1:1 sequential BW.
  double ideal_ring = 0;
  for (int i = 0; i < ngpu; i++)
    ideal_ring += seq_bw[i][(i + 1) % ngpu];

  printf("\n");
  printf("===========================================================\n");
  printf("  PCIe LINK SCORE:           %.2f\n", avg_per_gpu / PCIE_X16_THEORETICAL);
  printf("  (%.2f GB/s avg  /  %.1f GB/s PCIe 5.0 x16 theoretical)\n",
         avg_per_gpu, PCIE_X16_THEORETICAL);
  printf("\n");
  printf("  DENSE INTERCONNECT SCORE:  %.2f\n", total_ring / ideal_ring);
  printf("  (%.2f GB/s ring  /  %.2f GB/s ideal)\n", total_ring, ideal_ring);
  printf("\n");
  printf("  1.00 = perfect, 0.00 = none\n");
  printf("===========================================================\n");

  // Cleanup.
  CHECK(cudaFreeHost((void*)sw_flag));
  CHECK(cudaFreeHost((void*)fc_flag));
  for (int i = 0; i < ngpu; i++) {
    CHECK(cudaSetDevice(i));
    CHECK(cudaEventDestroy(fc_starts[i]));
    CHECK(cudaEventDestroy(fc_stops[i]));
    CHECK(cudaFree(sw_src[i]));
    for (int r = 0; r < ngpu - 1; r++) {
      int peer = (i + r + 1) % ngpu;
      CHECK(cudaSetDevice(peer));
      CHECK(cudaFree(sw_dst[i][r]));
      CHECK(cudaSetDevice(i));
      CHECK(cudaStreamDestroy(fc_streams[i][r]));
    }
  }

  return 0;
}

// ---- Latency mode ----

// Get the SM clock rate in kHz for a given device.
static double get_clock_rate_khz(int dev) {
  int clock_khz;
  CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, dev));
  return (double)clock_khz;
}

// Convert clock64() cycle delta to microseconds.
static double cycles_to_us(long long cycles, double clock_khz) {
  return (double)cycles / (clock_khz * 1000.0) * 1e6;
}

// Measure latency of a single remote read from reader GPU reading src_buf
// on another GPU. Returns average latency in microseconds.
// `cycles_buf` must be a device pointer to a long long on the reader GPU.
static double measure_latency(int reader, void* src_buf, void* local_buf,
                              void* cycles_buf, double clock_khz,
                              cudaStream_t stream) {
  latency_read_kernel<<<1, 1, 0, stream>>>(
      (const int*)src_buf, (int*)local_buf,
      LATENCY_WORDS, LATENCY_ITERS, (long long*)cycles_buf);
  CHECK(cudaStreamSynchronize(stream));

  long long cycles;
  CHECK(cudaMemcpy(&cycles, cycles_buf, sizeof(long long), cudaMemcpyDeviceToHost));
  return cycles_to_us(cycles, clock_khz) / LATENCY_ITERS / LATENCY_WORDS;
}

static void run_latency_tests(int ngpu) {
  printf("=== LATENCY MODE (%zu-byte remote reads, %d iterations) ===\n\n", LATENCY_SIZE, LATENCY_ITERS);

  // Allocate one source buffer, local scratch, and cycles buffer per GPU.
  std::vector<void*> bufs(ngpu), local(ngpu), cycles_buf(ngpu);
  std::vector<cudaStream_t> streams(ngpu);
  std::vector<double> clock_khz(ngpu);
  for (int i = 0; i < ngpu; i++) {
    CHECK(cudaSetDevice(i));
    CHECK(cudaMalloc(&bufs[i], LATENCY_SIZE));
    CHECK(cudaMemset(bufs[i], i, LATENCY_SIZE));
    CHECK(cudaMalloc(&local[i], sizeof(int)));
    CHECK(cudaMalloc(&cycles_buf[i], sizeof(long long)));
    CHECK(cudaStreamCreate(&streams[i]));
    clock_khz[i] = get_clock_rate_khz(i);
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
      double us = measure_latency(r, bufs[s], local[r], cycles_buf[r], clock_khz[r], streams[r]);
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

    std::barrier round_barrier(ngpu);
    std::vector<double> per_gpu_lat(ngpu);
    std::vector<std::thread> round_threads;

    for (int i = 0; i < ngpu; i++) {
      round_threads.emplace_back([&, i, offset]() {
        int peer = (i + offset) % ngpu;
        CHECK(cudaSetDevice(i));

        round_barrier.arrive_and_wait();

        latency_read_kernel<<<1, 1, 0, streams[i]>>>(
            (const int*)bufs[peer], (int*)local[i],
            LATENCY_WORDS, LATENCY_ITERS, (long long*)cycles_buf[i]);
        CHECK(cudaStreamSynchronize(streams[i]));

        long long cyc;
        CHECK(cudaMemcpy(&cyc, cycles_buf[i], sizeof(long long), cudaMemcpyDeviceToHost));
        per_gpu_lat[i] = cycles_to_us(cyc, clock_khz[i]) / LATENCY_ITERS / LATENCY_WORDS;
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
  }

  // ---- Test 2b: Single reader, all peers concurrent ----
  printf("\n=== Single reader, all %d peers concurrent (us) ===\n\n", ngpu - 1);

  std::vector<double> sr_lat(ngpu);

  for (int reader = 0; reader < ngpu; reader++) {
    CHECK(cudaSetDevice(reader));
    std::vector<cudaStream_t> peer_streams(ngpu - 1);
    std::vector<void*> peer_local(ngpu - 1), peer_cycles(ngpu - 1);
    for (int r = 0; r < ngpu - 1; r++) {
      CHECK(cudaStreamCreate(&peer_streams[r]));
      CHECK(cudaMalloc(&peer_local[r], sizeof(int)));
      CHECK(cudaMalloc(&peer_cycles[r], sizeof(long long)));
    }

    // Warmup.
    for (int r = 0; r < ngpu - 1; r++) {
      int peer = (reader + r + 1) % ngpu;
      latency_read_kernel<<<1, 1, 0, peer_streams[r]>>>(
          (const int*)bufs[peer], (int*)peer_local[r],
          LATENCY_WORDS, 1, (long long*)peer_cycles[r]);
    }
    for (int r = 0; r < ngpu - 1; r++)
      CHECK(cudaStreamSynchronize(peer_streams[r]));

    // Timed: wall-clock from first launch to all done.
    cudaEvent_t ev_start, ev_stop;
    CHECK(cudaEventCreate(&ev_start));
    CHECK(cudaEventCreate(&ev_stop));
    CHECK(cudaEventRecord(ev_start, peer_streams[0]));
    for (int r = 0; r < ngpu - 1; r++) {
      int peer = (reader + r + 1) % ngpu;
      latency_read_kernel<<<1, 1, 0, peer_streams[r]>>>(
          (const int*)bufs[peer], (int*)peer_local[r],
          LATENCY_WORDS, LATENCY_ITERS, (long long*)peer_cycles[r]);
    }
    for (int r = 1; r < ngpu - 1; r++)
      CHECK(cudaStreamSynchronize(peer_streams[r]));
    CHECK(cudaEventRecord(ev_stop, peer_streams[0]));
    CHECK(cudaStreamSynchronize(peer_streams[0]));

    float ms;
    CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
    sr_lat[reader] = (double)ms * 1000.0 / LATENCY_ITERS / LATENCY_WORDS;
    printf("GPU %d: %.2f us\n", reader, sr_lat[reader]);

    CHECK(cudaEventDestroy(ev_start));
    CHECK(cudaEventDestroy(ev_stop));
    for (int r = 0; r < ngpu - 1; r++) {
      CHECK(cudaFree(peer_local[r]));
      CHECK(cudaFree(peer_cycles[r]));
      CHECK(cudaStreamDestroy(peer_streams[r]));
    }
  }

  // ---- Test 3: All GPUs read all peers simultaneously ----
  printf("\n=== All GPUs read all peers simultaneously (us) ===\n");
  printf("Each GPU has %d streams, %d total concurrent kernels.\n\n",
         ngpu - 1, ngpu * (ngpu - 1));

  // Allocate per-GPU, per-peer streams, scratch, and cycle buffers.
  std::vector<std::vector<cudaStream_t>> ac_streams(ngpu, std::vector<cudaStream_t>(ngpu - 1));
  std::vector<std::vector<void*>> ac_local(ngpu, std::vector<void*>(ngpu - 1));
  std::vector<std::vector<void*>> ac_cycles(ngpu, std::vector<void*>(ngpu - 1));
  for (int i = 0; i < ngpu; i++) {
    CHECK(cudaSetDevice(i));
    for (int r = 0; r < ngpu - 1; r++) {
      CHECK(cudaStreamCreate(&ac_streams[i][r]));
      CHECK(cudaMalloc(&ac_local[i][r], sizeof(int)));
      CHECK(cudaMalloc(&ac_cycles[i][r], sizeof(long long)));
    }
  }

  // Warmup.
  for (int i = 0; i < ngpu; i++) {
    CHECK(cudaSetDevice(i));
    for (int r = 0; r < ngpu - 1; r++) {
      int peer = (i + r + 1) % ngpu;
      latency_read_kernel<<<1, 1, 0, ac_streams[i][r]>>>(
          (const int*)bufs[peer], (int*)ac_local[i][r],
          LATENCY_WORDS, 1, (long long*)ac_cycles[i][r]);
    }
  }
  for (int i = 0; i < ngpu; i++) {
    CHECK(cudaSetDevice(i));
    for (int r = 0; r < ngpu - 1; r++)
      CHECK(cudaStreamSynchronize(ac_streams[i][r]));
  }

  // Timed: wall-clock per GPU via cudaEvent.
  std::barrier ac_barrier(ngpu);
  std::vector<double> ac_lat(ngpu);
  std::vector<std::thread> ac_threads;

  for (int i = 0; i < ngpu; i++) {
    ac_threads.emplace_back([&, i]() {
      CHECK(cudaSetDevice(i));

      cudaEvent_t ev_start, ev_stop;
      CHECK(cudaEventCreate(&ev_start));
      CHECK(cudaEventCreate(&ev_stop));

      ac_barrier.arrive_and_wait();

      CHECK(cudaEventRecord(ev_start, ac_streams[i][0]));
      for (int r = 0; r < ngpu - 1; r++) {
        int peer = (i + r + 1) % ngpu;
        latency_read_kernel<<<1, 1, 0, ac_streams[i][r]>>>(
            (const int*)bufs[peer], (int*)ac_local[i][r],
            LATENCY_WORDS, LATENCY_ITERS, (long long*)ac_cycles[i][r]);
      }
      for (int r = 1; r < ngpu - 1; r++)
        CHECK(cudaStreamSynchronize(ac_streams[i][r]));
      CHECK(cudaEventRecord(ev_stop, ac_streams[i][0]));
      CHECK(cudaStreamSynchronize(ac_streams[i][0]));

      float ms;
      CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
      ac_lat[i] = (double)ms * 1000.0 / LATENCY_ITERS / LATENCY_WORDS;

      CHECK(cudaEventDestroy(ev_start));
      CHECK(cudaEventDestroy(ev_stop));
    });
  }
  for (auto& t : ac_threads) t.join();

  double ac_avg = 0;
  for (int i = 0; i < ngpu; i++) {
    printf("GPU %d: %.2f us (max across %d peers)\n", i, ac_lat[i], ngpu - 1);
    ac_avg += ac_lat[i];
  }
  ac_avg /= ngpu;

  // Find min from sequential NxN.
  double seq_min = 1e9;
  for (int i = 0; i < ngpu; i++)
    for (int j = 0; j < ngpu; j++)
      if (i != j) seq_min = std::min(seq_min, seq_lat[i][j]);

  double ac_max = 0;
  for (int i = 0; i < ngpu; i++)
    ac_max = std::max(ac_max, ac_lat[i]);

  printf("\n");
  printf("===========================================================\n");
  printf("  Min latency:           %6.2f us  (best pair, isolated)\n", seq_min);
  printf("  Mean latency:          %6.2f us  (per GPU under full load)\n", ac_avg);
  printf("\n");
  printf("  EFFECTIVE LATENCY:     %6.2f us  (all GPUs done reading all peers)\n", ac_max);
  printf("===========================================================\n");

  // Cleanup.
  for (int i = 0; i < ngpu; i++) {
    CHECK(cudaSetDevice(i));
    for (int r = 0; r < ngpu - 1; r++) {
      CHECK(cudaFree(ac_local[i][r]));
      CHECK(cudaFree(ac_cycles[i][r]));
      CHECK(cudaStreamDestroy(ac_streams[i][r]));
    }
  }
  for (int i = 0; i < ngpu; i++) {
    CHECK(cudaSetDevice(i));
    CHECK(cudaFree(bufs[i]));
    CHECK(cudaFree(local[i]));
    CHECK(cudaFree(cycles_buf[i]));
    CHECK(cudaStreamDestroy(streams[i]));
  }
}

// ---- Allreduce mode ----

// Benchmark custom allreduce via CUDA graph replay. Returns time in us.
static double bench_custom_ar(int ngpu, size_t sz,
                              std::vector<void*>& input, std::vector<void*>& output,
                              std::vector<pcie_ar::Signal*>& sigs,
                              std::vector<void*>& rd_dev, pcie_ar::RankSignals& rs,
                              std::vector<cudaStream_t>& streams) {
  using T = half;
  constexpr int d = pcie_ar::packed_t<T>::P::size;
  int num_elements = sz / sizeof(T);
  int packed_size = num_elements / d;
  int threads = 512;
  int blocks = std::min(36, (packed_size + threads - 1) / threads);
  blocks = std::max(blocks, 1);
  int iters = (sz <= 1024 * 1024) ? 2000 : 200;

  // Reset signals.
  for (int i = 0; i < ngpu; i++) {
    CHECK(cudaSetDevice(i));
    CHECK(cudaMemset(sigs[i], 0, sizeof(pcie_ar::Signal)));
    CHECK(cudaDeviceSynchronize());
  }

  // Warmup (non-graph, establishes barrier state).
  {
    std::barrier bar(ngpu);
    std::vector<std::thread> tv;
    for (int i = 0; i < ngpu; i++) {
      tv.emplace_back([&, i]() {
        CHECK(cudaSetDevice(i));
        bar.arrive_and_wait();
        for (int w = 0; w < 20; w++)
          pcie_ar::launch_kernel<T>(ngpu, blocks, threads, streams[i],
              (pcie_ar::RankData*)rd_dev[i], rs, sigs[i], (T*)output[i], i, packed_size);
        CHECK(cudaStreamSynchronize(streams[i]));
      });
    }
    for (auto& t : tv) t.join();
  }

  // Capture one allreduce per GPU into a graph.
  std::vector<cudaGraph_t> graphs(ngpu);
  std::vector<cudaGraphExec_t> execs(ngpu);
  for (int i = 0; i < ngpu; i++) {
    CHECK(cudaSetDevice(i));
    CHECK(cudaStreamBeginCapture(streams[i], cudaStreamCaptureModeThreadLocal));
    pcie_ar::launch_kernel<T>(ngpu, blocks, threads, streams[i],
        (pcie_ar::RankData*)rd_dev[i], rs, sigs[i], (T*)output[i], i, packed_size);
    CHECK(cudaStreamEndCapture(streams[i], &graphs[i]));
    CHECK(cudaGraphInstantiate(&execs[i], graphs[i], 0));
  }

  // Timed: replay graphs from concurrent threads.
  std::barrier tbar(ngpu);
  std::vector<double> per_gpu_us(ngpu);
  std::vector<std::thread> tv;

  for (int i = 0; i < ngpu; i++) {
    tv.emplace_back([&, i]() {
      CHECK(cudaSetDevice(i));
      cudaEvent_t e0, e1;
      CHECK(cudaEventCreate(&e0));
      CHECK(cudaEventCreate(&e1));
      tbar.arrive_and_wait();
      CHECK(cudaEventRecord(e0, streams[i]));
      for (int it = 0; it < iters; it++)
        CHECK(cudaGraphLaunch(execs[i], streams[i]));
      CHECK(cudaEventRecord(e1, streams[i]));
      CHECK(cudaStreamSynchronize(streams[i]));
      float ms;
      CHECK(cudaEventElapsedTime(&ms, e0, e1));
      per_gpu_us[i] = (double)ms * 1000.0 / iters;
      CHECK(cudaEventDestroy(e0));
      CHECK(cudaEventDestroy(e1));
    });
  }
  for (auto& t : tv) t.join();

  for (int i = 0; i < ngpu; i++) {
    CHECK(cudaGraphExecDestroy(execs[i]));
    CHECK(cudaGraphDestroy(graphs[i]));
  }

  double max_us = 0;
  for (int i = 0; i < ngpu; i++) max_us = std::max(max_us, per_gpu_us[i]);
  return max_us;
}

// Benchmark NCCL allreduce (no graph — NCCL group ops don't support
// multi-stream capture in single-process mode). Returns time in us.
static double bench_nccl_ar(int ngpu, size_t sz,
                            std::vector<void*>& input, std::vector<void*>& output,
                            std::vector<ncclComm_t>& comms,
                            std::vector<cudaStream_t>& streams) {
  int count = sz / sizeof(half);
  int iters = (sz <= 1024 * 1024) ? 2000 : 200;

  // Warmup.
  for (int w = 0; w < 20; w++) {
    CHECK_NCCL(ncclGroupStart());
    for (int i = 0; i < ngpu; i++)
      CHECK_NCCL(ncclAllReduce(input[i], output[i], count, ncclFloat16, ncclSum, comms[i], streams[i]));
    CHECK_NCCL(ncclGroupEnd());
  }
  for (int i = 0; i < ngpu; i++) {
    CHECK(cudaSetDevice(i));
    CHECK(cudaStreamSynchronize(streams[i]));
  }

  // Timed.
  std::vector<cudaEvent_t> e0(ngpu), e1(ngpu);
  for (int i = 0; i < ngpu; i++) {
    CHECK(cudaSetDevice(i));
    CHECK(cudaEventCreate(&e0[i]));
    CHECK(cudaEventCreate(&e1[i]));
    CHECK(cudaEventRecord(e0[i], streams[i]));
  }
  for (int it = 0; it < iters; it++) {
    CHECK_NCCL(ncclGroupStart());
    for (int i = 0; i < ngpu; i++)
      CHECK_NCCL(ncclAllReduce(input[i], output[i], count, ncclFloat16, ncclSum, comms[i], streams[i]));
    CHECK_NCCL(ncclGroupEnd());
  }
  double max_us = 0;
  for (int i = 0; i < ngpu; i++) {
    CHECK(cudaSetDevice(i));
    CHECK(cudaEventRecord(e1[i], streams[i]));
    CHECK(cudaStreamSynchronize(streams[i]));
    float ms;
    CHECK(cudaEventElapsedTime(&ms, e0[i], e1[i]));
    double us = (double)ms * 1000.0 / iters;
    max_us = std::max(max_us, us);
    CHECK(cudaEventDestroy(e0[i]));
    CHECK(cudaEventDestroy(e1[i]));
  }
  return max_us;
}

static void run_allreduce_tests(int ngpu) {
  using T = half;
  constexpr int d = pcie_ar::packed_t<T>::P::size;

  if (ngpu < 2 || ngpu > 8) {
    fprintf(stderr, "Allreduce mode requires 2-8 GPUs, got %d\n", ngpu);
    return;
  }

  printf("=== ALLREDUCE MODE (%d GPUs, fp16) ===\n\n", ngpu);

  std::vector<size_t> sizes;
  for (size_t s = 256; s <= 64ULL * 1024 * 1024; s *= 2) {
    sizes.push_back(s);
    if (s >= 32 * 1024 && s < 64 * 1024) {
      sizes.push_back(s + s / 4);   // +25%
      sizes.push_back(s + s / 2);   // +50%
      sizes.push_back(s + 3 * s / 4); // +75%
    }
  }

  size_t max_sz = sizes.back();

  // Per-GPU resources (shared between custom and NCCL benchmarks).
  std::vector<void*> input(ngpu), output(ngpu);
  std::vector<pcie_ar::Signal*> sigs(ngpu);
  std::vector<void*> rd_dev(ngpu);
  std::vector<cudaStream_t> streams(ngpu);

  for (int i = 0; i < ngpu; i++) {
    CHECK(cudaSetDevice(i));
    CHECK(cudaMalloc(&input[i], max_sz));
    CHECK(cudaMalloc(&output[i], max_sz));
    CHECK(cudaMemset(input[i], i + 1, max_sz));
    CHECK(cudaMalloc((void**)&sigs[i], sizeof(pcie_ar::Signal)));
    CHECK(cudaMemset(sigs[i], 0, sizeof(pcie_ar::Signal)));
    CHECK(cudaStreamCreate(&streams[i]));
  }

  // Custom allreduce setup.
  pcie_ar::RankData rd;
  for (int i = 0; i < ngpu; i++) rd.ptrs[i] = input[i];
  for (int i = 0; i < ngpu; i++) {
    CHECK(cudaSetDevice(i));
    CHECK(cudaMalloc(&rd_dev[i], sizeof(pcie_ar::RankData)));
    CHECK(cudaMemcpy(rd_dev[i], &rd, sizeof(pcie_ar::RankData), cudaMemcpyHostToDevice));
  }
  pcie_ar::RankSignals rs;
  for (int i = 0; i < ngpu; i++) rs.signals[i] = sigs[i];

  // NCCL setup.
  std::vector<ncclComm_t> comms(ngpu);
  std::vector<int> devs(ngpu);
  for (int i = 0; i < ngpu; i++) devs[i] = i;
  CHECK_NCCL(ncclCommInitAll(comms.data(), ngpu, devs.data()));

  printf("%10s  %12s  %12s  %s\n", "Size", "Custom (us)", "NCCL (us)", "Winner");
  printf("---------- ------------ ------------ --------\n");

  for (size_t sz : sizes) {
    int num_elements = sz / sizeof(T);
    if (num_elements % d != 0) continue;

    double custom_us = bench_custom_ar(ngpu, sz, input, output, sigs, rd_dev, rs, streams);
    double nccl_us = bench_nccl_ar(ngpu, sz, input, output, comms, streams);

    char sz_str[32];
    if (sz >= 1024 * 1024) snprintf(sz_str, sizeof(sz_str), "%zu MB", sz / (1024 * 1024));
    else if (sz >= 1024)   snprintf(sz_str, sizeof(sz_str), "%zu KB", sz / 1024);
    else                   snprintf(sz_str, sizeof(sz_str), "%zu B", sz);

    const char* winner = (custom_us < nccl_us) ? "custom" : "NCCL";
    double ratio = (custom_us < nccl_us) ? nccl_us / custom_us : custom_us / nccl_us;
    printf("%10s  %12.1f  %12.1f  %s (%.1fx)\n", sz_str, custom_us, nccl_us, winner, ratio);
  }

  // Cleanup.
  for (int i = 0; i < ngpu; i++) CHECK_NCCL(ncclCommDestroy(comms[i]));
  for (int i = 0; i < ngpu; i++) {
    CHECK(cudaSetDevice(i));
    CHECK(cudaFree(input[i]));
    CHECK(cudaFree(output[i]));
    CHECK(cudaFree(sigs[i]));
    CHECK(cudaFree(rd_dev[i]));
    CHECK(cudaStreamDestroy(streams[i]));
  }
}
