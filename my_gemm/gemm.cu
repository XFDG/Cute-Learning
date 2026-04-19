#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <cute/tensor.hpp>

#include "utils.h"

// 当前实验先以 half 精度为主，后面如果要切到 bf16 / float，
// 优先改这里，再逐步检查 MMA 配置和参考实现。
using T = cute::half_t;
using namespace cute;

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define OFFSETCOL(row, col, ld) ((col) * (ld) + (row))

template <typename Element>
void cpu_f16_gemm_tn(const Element* a, const Element* b, Element* c, int m, int n, int k) {
  // CPU 参考实现：
  // A 按 (M, K) 行主序读取，B 按转置视角访问，
  // 用来做 correctness 对比，而不是追求性能。
  //
  // 注意：这里是 float 累加、最后再 cast 回 half。
  // 如果 GPU 侧使用的是 half accumulate，那么两边不是完全同口径比较。
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      float acc = 0.0f;
      for (int kk = 0; kk < k; ++kk) {
        acc += static_cast<float>(a[OFFSET(row, kk, k)]) *
               static_cast<float>(b[OFFSETCOL(kk, col, k)]);
      }
      c[OFFSET(row, col, n)] = static_cast<Element>(acc);
    }
  }
}

template <typename Element, int BM, int BN, int BK, typename TiledMMA>
__global__ void my_first_gemm_kernel(const Element* a_ptr,
                                     const Element* b_ptr,
                                     Element* c_ptr,
                                     int m,
                                     int n,
                                     int k) {
  // 1. 先把原始指针包装成 CuTe Tensor。
  // A: (M, K)
  // B: 这里按照 (N, K) 来看，和下面 TN 的 MMA 配置保持一致。
  // C: (M, N)
  Tensor A = make_tensor(make_gmem_ptr(a_ptr), make_shape(m, k), make_stride(k, Int<1>{}));
  //   A 形状 M*K row-major ，索引方式 A(row, col) = A[row * k + col * 1]  
  // 第一维步长 k：运行时才知道 ，第二维步长 1：编译期固定就是 1 写法： Int<1>{}
  Tensor B = make_tensor(make_gmem_ptr(b_ptr), make_shape(n, k), make_stride(k, Int<1>{}));
  // B 形状 N*K row-major （按转置视角），索引方式 B(row, col) = B[row * k + col * 1]
  Tensor C = make_tensor(make_gmem_ptr(c_ptr), make_shape(m, n), make_stride(n, Int<1>{}));
  // C 形状 M*N row-major，索引方式 C(row, col) = C[row * n + col * 1]   C(i, j) -> c_ptr[i * n + j *1]


  // 2. 每个 CTA 只负责 C 的一个   
  // 同时取出本 CTA 对应的 A / B tile。
  Tensor gA = local_tile(A, make_tile(Int<BM>{}, Int<BK>{}), make_coord(blockIdx.y, _));
  Tensor gB = local_tile(B, make_tile(Int<BN>{}, Int<BK>{}), make_coord(blockIdx.x, _));
  Tensor gC = local_tile(C, make_tile(Int<BM>{}, Int<BN>{}), make_coord(blockIdx.y, blockIdx.x));

  TiledMMA tiled_mma; 
  //创建一个 TiledMMA 对象，后续所有切分和计算都基于它来做。这个对象本身不占用资源，里面的配置参数也都是编译期常量。
  auto thr_mma = tiled_mma.get_slice(threadIdx.x);
  //从整个 tiled_mma 里，取出 当前线程 threadIdx.x 对应的那一份视角。

  //get_slice(threadIdx.x) 之后，再通过 partition_A/B/C 把 A/B/C 的线程分区应用到对应 Tensor 上；
  // 这些结果 Tensor 的第一维 MMA 表示一条 MMA 指令一次会消费的元素集合。

  // 3. 把 CTA tile 再切给每个线程。
  // tAgA / tBgB / tCgC 描述的是“这个线程该处理哪些元素”。
  // tAgA  thread A global A
  auto tAgA = thr_mma.partition_A(gA);
    // 当前线程该拿 A 的哪一小份
  auto tBgB = thr_mma.partition_B(gB);
  auto tCgC = thr_mma.partition_C(gC);

  // 4. 为本线程分配寄存器片段。
  // tArA / tBrB 存输入碎片，tCrC 存累加结果。
  // tArA  thread A register A
  auto tArA = thr_mma.partition_fragment_A(gA(_, _, 0));
  //make_fragment_C(...)：给当前线程分配对应的寄存器累加器 fragment
  auto tBrB = thr_mma.partition_fragment_B(gB(_, _, 0));
  auto tCrC = thr_mma.partition_fragment_C(gC(_, _));

  // 5. 累加器清零，准备进入 K 方向主循环。
  clear(tCrC);

  int num_k_tiles = size<2>(gA);
#pragma unroll 1
  for (int tile_k = 0; tile_k < num_k_tiles; ++tile_k) {
    // 6. 当前版本先走最简单路径：
    // 直接从 global memory 拷到寄存器，再做 mma。
    // 这是最容易读懂和验证的 baseline。
    cute::copy(tAgA(_, _, _, tile_k), tArA);
    cute::copy(tBgB(_, _, _, tile_k), tBrB);
    //---------------------------------真正的gemm计算------------------------------------------------------
    cute::gemm(tiled_mma, tCrC, tArA, tBrB, tCrC);
  }

  // 7. 把寄存器累加结果写回全局内存。
  cute::copy(tCrC, tCgC);
}

template <typename Element>
void launch_my_first_gemm(Element* d_a, Element* d_b, Element* d_c, int m, int n, int k) {
  // 这里先复用一个最基础的 SM80 half Tensor Core MMA 原语。
  // 在 Blackwell / Ada / Ampere 上通常都能编译运行，
  // 但这不代表它已经是当前 GPU 的最优写法。
  //
  // 这个原语本身是 F16 输入、F16 累加、F16 输出。
  // 所以后面如果 correctness 误差偏大，先检查这里和参考实现是否口径一致。
  using MMA = decltype(make_tiled_mma(
      SM80_16x8x16_F16F16F16F16_TN{},
      Layout<Shape<_1, _1, _1>>{}));

  // 先固定一个容易调试的 tile 大小。
  // 后续调优时，优先改这一组参数。
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 32;

  dim3 block(size(MMA{}));
  dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);

  // host 侧只负责 launch，不在这里混进测试逻辑。
  my_first_gemm_kernel<Element, BM, BN, BK, MMA><<<grid, block>>>(d_a, d_b, d_c, m, n, k);
  CHECK_CUDA(cudaGetLastError());
}

template <typename Element>
float run_correctness_check(int m, int n, int k) {
  // correctness 阶段：
  // 1. 生成输入
  // 2. 用 CPU 参考实现算一份
  // 3. 跑 GPU kernel
  // 4. 比较最大误差
  size_t size_a = static_cast<size_t>(m) * k * sizeof(Element);
  size_t size_b = static_cast<size_t>(k) * n * sizeof(Element);
  size_t size_c = static_cast<size_t>(m) * n * sizeof(Element);

  Element* h_a = static_cast<Element*>(std::malloc(size_a));
  Element* h_b = static_cast<Element*>(std::malloc(size_b));
  Element* h_ref = static_cast<Element*>(std::malloc(size_c));
  Element* h_out = static_cast<Element*>(std::malloc(size_c));

  Element* d_a = nullptr;
  Element* d_b = nullptr;
  Element* d_c = nullptr;

  CHECK_CUDA(cudaMalloc(&d_a, size_a));
  CHECK_CUDA(cudaMalloc(&d_b, size_b));
  CHECK_CUDA(cudaMalloc(&d_c, size_c));

  // 这里固定随机种子，方便复现实验。
  std::srand(0);
  for (int i = 0; i < m * k; ++i) {
    h_a[i] = static_cast<Element>(std::rand() / static_cast<float>(RAND_MAX));
  }
  for (int i = 0; i < k * n; ++i) {
    h_b[i] = static_cast<Element>(std::rand() / static_cast<float>(RAND_MAX));
  }

  cpu_f16_gemm_tn(h_a, h_b, h_ref, m, n, k);
  // h_ref 现在存着 CPU 算出来的结果，用来和 GPU 输出做对比。

  CHECK_CUDA(cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice));

  launch_my_first_gemm(d_a, d_b, d_c, m, n, k);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(h_out, d_c, size_c, cudaMemcpyDeviceToHost));
  // h_out 现在存着 GPU 算出来的结果。

  float max_error = 0.0f;
  for (int i = 0; i < m * n; ++i) {
    float diff = std::fabs(static_cast<float>(h_out[i]) - static_cast<float>(h_ref[i]));
    // 这里简单用绝对误差来衡量结果正确性，后续如果需要更细致的分析，可以改成相对误差，或者输出误差分布等。
    if (diff > max_error) {
      printf("Mismatch at index %d: GPU = %f, CPU = %f, Diff = %f\n", i, static_cast<float>(h_out[i]), static_cast<float>(h_ref[i]), diff);
      max_error = diff;
    }
  }

  std::free(h_a);
  std::free(h_b);
  std::free(h_ref);
  std::free(h_out);
  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_c));

  return max_error;
}

template <typename Element>
float run_perf_test(int m, int n, int k, int repeat) {
  // perf 阶段：
  // 只关注 kernel 平均时间，不再做 CPU 对比。
  size_t size_a = static_cast<size_t>(m) * k * sizeof(Element);
  size_t size_b = static_cast<size_t>(k) * n * sizeof(Element);
  size_t size_c = static_cast<size_t>(m) * n * sizeof(Element);

  Element* d_a = nullptr;
  Element* d_b = nullptr;
  Element* d_c = nullptr;

  CHECK_CUDA(cudaMalloc(&d_a, size_a));
  CHECK_CUDA(cudaMalloc(&d_b, size_b));
  CHECK_CUDA(cudaMalloc(&d_c, size_c));

  cudaEvent_t start;
  cudaEvent_t stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  // 用 CUDA Event 统计平均执行时间。
  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < repeat; ++i) {
    launch_my_first_gemm(d_a, d_b, d_c, m, n, k);
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float elapsed_ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_c));

  return elapsed_ms / 1000.0f / repeat;
}

int main() {
  CHECK_CUDA(cudaSetDevice(0));

  // 第一阶段输出：方便确认当前跑的是哪份实验代码。
  std::printf("algo = my_first_cute_gemm\n");

  // 先做一个小尺寸正确性测试。
  int check_m = 256;
  int check_n = 256;
  int check_k = 256;
  float max_error = run_correctness_check<T>(check_m, check_n, check_k);
  std::printf("Correctness: M=%d N=%d K=%d Max Error = %.6f\n", check_m, check_n, check_k, max_error);

  // 再做一个固定尺寸的性能测试。
  // 这一步的结果主要用于横向比较不同实现版本。
  int perf_m = 2048;
  int perf_n = 2048;
  int perf_k = 2048;
  int repeat = 20;
  float avg_sec = run_perf_test<T>(perf_m, perf_n, perf_k, repeat);
  double gflops = static_cast<double>(perf_m) * perf_n * perf_k * 2.0 / (1024.0 * 1024.0 * 1024.0) / avg_sec;
  std::printf("Performance: M=%d N=%d K=%d Time = %.6f s, GFLOPS = %.2f\n",
              perf_m, perf_n, perf_k, avg_sec, gflops);

  return 0;
}
