# CuTe GEMM 实验报告

## 开发环境

- 操作系统：`Linux 6.6.87.2-microsoft-standard-WSL2 x86_64 GNU/Linux`
- GPU：`NVIDIA GeForce RTX 5060 Laptop GPU`
- NVIDIA Driver：`591.74`
- CUDA Toolkit：`13.1`
- `nvcc` 版本：`V13.1.115`
- 当前目标架构：`sm_120`
- 当前构建目标：`TARGET=gemm`，`SRC=gemm.cu`
- CUTLASS 路径：`/home/ai/workspace/Cute-Learning/flashdecoding/src/cutlass`

## 仓库信息

- 我的 fork 仓库：https://github.com/XFDG/Cute-Learning
- 上游仓库：https://github.com/DD-DuDa/Cute-Learning
- 当前开发分支：`feat/my-gemm-baseline`
- 当前开发目录：`/home/ai/workspace/Cute-Learning/my_gemm`
- 当前已推送提交：`c31c93a`

日期：`2026-04-19`

## 1. 实验目的

今天这次开发的目标不是直接把 GEMM 算子优化到最好，而是先在 `my_gemm` 目录里建立一份自己能真正看懂的 CuTe GEMM baseline。

重点不是“先卷性能”，而是：

1. 看懂 CuTe 里 Tensor 是怎么描述的。
2. 看懂一个 block / 一个线程分别负责什么。
3. 看懂 `make_tensor()`、`local_tile()`、`partition_*()`、`cute::gemm()` 这些语法在 GEMM 里各自扮演的角色。
4. 建立一个后续可以继续手写官方教程版本的练习入口。

## 2. 今天完成了什么

今天围绕 `my_gemm` 目录完成了下面这些事情：

1. 保留并验证了已有的 baseline 文件 `gemm.cu` 可以编译和运行。
2. 让工程依赖只指向 `Cute-Learning` 仓库内的 CUTLASS，而不是依赖 `LLMQRT` 里的副本。
3. 新建了 `tutorial_gemm_scratch.cu` 作为后续按 NVIDIA 官方教程手写 CuTe GEMM 的空白练习文件。
4. 继续观察 correctness 与 performance 输出，并开始打印部分 mismatch 样本。
5. 将开发背景、常用命令、工具链和调试结论记录进 `DEV_NOTES.md`。

## 3. 当前工程入口

当前主要相关文件如下：

- `gemm.cu`
  当前可编译运行的 CuTe GEMM baseline
- `tutorial_gemm_scratch.cu`
  后续按官方教程自己手写的实验文件
- `utils.h`
  CUDA 错误检查等辅助宏
- `Makefile`
  调用仓库根目录 `common.mk` 的构建入口
- `DEV_NOTES.md`
  过程记录、调试记录、工具命令

## 4. 当前 baseline 的执行链

现在这份 `gemm.cu` 的实际执行顺序是：

1. `main()`
   程序入口，先跑 correctness，再跑 performance。
2. `run_correctness_check()`
   分配 host/device 内存，准备输入，计算 CPU 参考结果，调用 GPU kernel，再把结果拷回 host。
3. `cpu_f16_gemm_tn()`
   CPU 参考实现，用来生成 `h_ref`。
4. `launch_my_first_gemm()`
   配置 `MMA`、`BM/BN/BK`、`grid/block`，然后发起 kernel。
5. `my_first_gemm_kernel()`
   真正的 CuTe kernel，完成 tile 划分、线程分工、copy、mma 累加、写回结果。

## 5. 今天重点关注的 CuTe 语法

下面这些语法是今天最核心的学习对象。

### 5.1 `make_tensor()`

作用：把原始指针包装成 CuTe Tensor。

在当前代码里：

```cpp
Tensor A = make_tensor(make_gmem_ptr(a_ptr), make_shape(m, k), make_stride(k, Int<1>{}));
```

它表达的不是“立刻搬数据”，而是：

1. 底层数据在哪里：`a_ptr`
2. 这块数据怎么看成几维：`(m, k)`
3. 每一维的步长是多少：`(k, 1)`

换句话说，`make_tensor()` 是在告诉 CuTe：

- 这是一块二维矩阵数据
- 它的逻辑形状是什么
- 它的索引规则是什么

### 5.2 `make_shape()`

作用：描述 Tensor 的逻辑形状。

例如：

```cpp
make_shape(m, k)
```

表示把数据看成一个 `(M, K)` 的二维对象。

### 5.3 `make_stride()`

作用：描述 Tensor 各维度的步长。

例如：

```cpp
make_stride(k, Int<1>{})
```

可以理解为行主序的 `(M, K)` 矩阵：

- 第一维前进一步，要跨 `k` 个元素
- 第二维前进一步，只跨 `1` 个元素

这就是为什么当前 `A(row, col)` 本质上对应：

```text
a_ptr[row * k + col]
```

### 5.4 `local_tile()`

作用：从整个全局 Tensor 里取出“当前 CTA 负责的那一块 tile”。

例如：

```cpp
Tensor gC = local_tile(C, make_tile(Int<BM>{}, Int<BN>{}), make_coord(blockIdx.y, blockIdx.x));
```

它的意思是：

1. 从整个 `C` 矩阵里
2. 取一个大小为 `BM x BN` 的小块
3. 这个小块的位置由 `blockIdx.y` 和 `blockIdx.x` 决定

所以 `local_tile()` 是 block 级别分工的核心语法。

### 5.5 `make_tile()`

作用：描述 tile 的大小。

例如：

```cpp
make_tile(Int<BM>{}, Int<BK>{})
```

就是告诉 CuTe：“我要按 `BM x BK` 的 tile 去切。”

这里用 `Int<...>{}` 是因为这些 tile 参数通常是编译期常量。

### 5.6 `make_coord()`

作用：告诉 CuTe 取哪一个 tile。

例如：

```cpp
make_coord(blockIdx.y, _)
```

这里可以读成：

- M 方向取第 `blockIdx.y` 个 tile
- K 方向先保留全部

其中 `_` 的意思很像“这一维先不要固定，后面再遍历”。

### 5.7 `make_tiled_mma()`

作用：定义 MMA 运算模型，也就是“线程组如何组织 MMA 指令”。

当前代码用的是：

```cpp
using MMA = decltype(make_tiled_mma(
    SM80_16x8x16_F16F16F16F16_TN{},
    Layout<Shape<_1, _1, _1>>{}));
```

这表示当前 baseline 选用的是一个基于 SM80 Tensor Core 的 MMA 原语。

这里最重要的理解不是把模板名背下来，而是知道：

- 这一步决定了底层用什么 MMA 原语
- 后续线程切分、fragment 形状、`cute::gemm()` 行为都依赖它

### 5.8 `get_slice()`

作用：从整个 `tiled_mma` 里，拿到“当前线程该负责的那一份视角”。

例如：

```cpp
auto thr_mma = tiled_mma.get_slice(threadIdx.x);
```

可以把它理解成：

- 整个 block 的 MMA 工作先定义好了
- 当前线程通过 `threadIdx.x` 去领自己那份任务

### 5.9 `partition_A/B/C()`

作用：把 block 级别的 tile 再细分到线程级别。

例如：

```cpp
auto tAgA = thr_mma.partition_A(gA);
auto tBgB = thr_mma.partition_B(gB);
auto tCgC = thr_mma.partition_C(gC);
```

这三句可以读成：

- 当前线程从 `gA` 里负责哪些输入元素
- 当前线程从 `gB` 里负责哪些输入元素
- 当前线程最终要写 `gC` 里的哪些输出元素

这就是为什么 CuTe kernel 里不一定能直接看到显式的 `row/col` 公式，因为线程分工已经被抽象到 `partition_*()` 里了。

### 5.10 `partition_fragment_A/B/C()`

作用：为当前线程分配寄存器 fragment。

例如：

```cpp
auto tArA = thr_mma.partition_fragment_A(gA(_, _, 0));
auto tBrB = thr_mma.partition_fragment_B(gB(_, _, 0));
auto tCrC = thr_mma.partition_fragment_C(gC(_, _));
```

可以理解为：

- `tArA`：当前线程的 A 寄存器输入片段
- `tBrB`：当前线程的 B 寄存器输入片段
- `tCrC`：当前线程的 C 寄存器累加器

### 5.11 `clear()`

作用：把 C 的寄存器累加器清零。

```cpp
clear(tCrC);
```

这对应普通 GEMM 里：

```text
acc = 0
```

### 5.12 `cute::copy()`

作用：做 Tensor 之间的数据拷贝。

在当前 baseline 里，它主要表示：

- 从全局内存 tile 拷到当前线程的寄存器 fragment
- 最后再从寄存器 fragment 拷回全局内存

所以 `copy()` 不是固定等于某一种物理指令，而是一个更高层的拷贝语义。

### 5.13 `cute::gemm()`

作用：对当前线程掌握的 fragment 做一次 GEMM 累加。

```cpp
cute::gemm(tiled_mma, tCrC, tArA, tBrB, tCrC);
```

可以把它理解成：

```text
tCrC = A_fragment * B_fragment + tCrC
```

它是当前 kernel 主循环里的核心计算语句。

## 6. 今天的运行结果

今天在老 baseline 上得到的调试输出是：

```text
algo = my_first_cute_gemm
Mismatch at index 0: GPU = 62.812500, CPU = 62.781250, Diff = 0.031250
Mismatch at index 1: GPU = 68.187500, CPU = 68.250000, Diff = 0.062500
Mismatch at index 235: GPU = 67.875000, CPU = 67.750000, Diff = 0.125000
Correctness: M=256 N=256 K=256 Max Error = 0.125000
Performance: M=2048 N=2048 K=2048 Time = 0.001734 s, GFLOPS = 9225.78
```

这说明：

1. 当前 baseline 能正常编译和运行。
2. 当前 CPU 参考结果和 GPU 输出不是完全一致。
3. 当前最大绝对误差是 `0.125000`。
4. 当前性能大约在 `9.2 TFLOPS`。

## 7. 对当前误差的理解

现在不能简单把 `Max Error = 0.125` 直接解释成“kernel 算错了”。

更合理的解释是：

1. GPU 路径用的是 `F16` 输入、`F16` 累加、`F16` 输出
2. CPU 参考实现是 `float` 累加、最后再 cast 回 `half`
3. 两边的数值口径并不完全一致

所以现在这份 baseline 更适合作为：

- CuTe 语法学习样本
- kernel 执行过程学习样本
- 调试入口

而不是拿来做严格的高精度数值基准。

## 8. 今天最重要的收获

今天最大的收获不是“性能提升了多少”，而是开始把 CuTe 的抽象对应到 GEMM 的实际执行过程上：

1. `make_tensor()` 负责描述“这块内存是什么形状、怎么索引”。
2. `local_tile()` 负责“把全局矩阵切到 CTA 级别”。
3. `get_slice()` 和 `partition_*()` 负责“把 CTA tile 再切到线程级别”。
4. `partition_fragment_*()` 负责“给线程分配寄存器片段”。
5. `cute::copy()` 和 `cute::gemm()` 负责“主循环里的搬运与计算”。

## 9. 未来计划

下一步更适合做下面这些事情：

1. 按 NVIDIA 官方教程，在 `tutorial_gemm_scratch.cu` 里从零手写最小版本。
2. 写的时候只保留最小骨架，不要一开始就加 correctness 和 benchmark。
3. 先把 `make_tensor()`、`local_tile()`、`partition_*()` 这条主线写顺。
4. 再回头对照当前 `gemm.cu`，看看哪些是你已经真正理解的，哪些还只是“能跑但没吃透”。
