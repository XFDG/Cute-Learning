# my_gemm 开发记录

## 仓库信息

- 我的 fork 仓库：https://github.com/XFDG/Cute-Learning
- 上游仓库：https://github.com/DD-DuDa/Cute-Learning
- 当前开发分支：`feat/my-gemm-baseline`
- 当前开发目录：`/home/ai/workspace/Cute-Learning/my_gemm`
- 当前已推送提交：`c31c93a`

## 目的

这个目录的目标不是直接写出最终最强版本，而是先做一个我们能完全看懂、能稳定调试、能逐步演进的 CuTe GEMM 实验目录。

当前阶段的目标是：

1. 先有一份能独立编译和运行的 baseline。
2. 先保证结果正确，再逐步优化性能。
3. 每次只改一个关键变量，避免一上来改动过大。

## 当前文件职责

- `Makefile`
  用来调用仓库根目录的 `common.mk`，复用统一的 CUDA / CUTLASS 构建逻辑。

- `gemm.cu`
  主实验文件，包含：
  - CuTe kernel
  - host 侧 launch
  - correctness check
  - performance test

- `utils.h`
  放调试和错误检查的小工具，避免把 `gemm.cu` 写乱。

## 开发格式约定

后续开发统一遵守下面这套格式：

1. 先保留一个可运行 baseline。
2. 每次改动尽量只动一类东西：
   - tile 参数
   - 数据类型
   - MMA 配置
   - 内存搬运方式
3. 每次改完都先跑 correctness，再看性能。
4. 新增优化前，先写清楚“为什么改”和“准备观察什么指标”。
5. 注释优先解释“这一段在做什么”和“为什么这样做”，不要写纯重复代码表面的注释。

## 建议的调试顺序

建议始终按下面顺序调试：

1. `make info`
   先确认编译架构、CUTLASS 路径和当前目标文件都对。

2. `make build`
   确认能编译通过。

3. `make run`
   先看 correctness 输出，再看性能输出。

4. 如果结果不对：
   - 先缩小尺寸
   - 再打印 tile / layout / fragment 的 shape
   - 最后再怀疑 MMA 配置或内存布局

## 当前观测到的现象

### 1. `make run` 会再次触发编译

这是因为仓库里的 `common.mk` 把 `build` 和 `run` 都声明成了 phony target，`run` 又依赖 `build`。

所以你执行：

```bash
make build
make run
```

时会看到编译输出重复一遍，这是正常现象，不是出错。

如果你只想直接运行已经编好的程序，可以手动执行：

```bash
./gemm
```

### 2. `compiler-bindir` 重复告警

日志里这条：

```text
nvcc warning : incompatible redefinition for option 'compiler-bindir'
```

说明当前 shell 环境里还有 Conda / NVCC 相关变量在额外注入 host compiler。

它现在没有阻塞编译和运行，所以暂时不是第一优先级问题。
等 baseline 稳定后，如果你愿意，再专门清理这部分环境。

### 3. CUTLASS 的 deprecated vector warning

像这些：

```text
long4 / ulong4 / double4 is deprecated
```

来自 CUTLASS 头文件和 CUDA 13.1 的组合。

这类 warning 对当前实验不是功能性错误，可以先忽略。

### 4. 当前运行结果

你这次的结果是：

- `DETECTED_GPU_ARCH=sm_120`
- `CUTLASS_ROOT` 已经正确找到
- baseline 可以正常编译和运行
- `Correctness: Max Error = 0.125000`
- `Performance: 2048^3 -> 9163.59 GFLOPS`

这说明：

1. 工程结构是通的。
2. 当前 kernel 是能跑的。
3. `Max Error = 0.125` 不一定说明 kernel 错了，更可能是比较口径还没对齐。

### 5. CUTLASS 依赖来源已经收口

这次还顺手修复了构建依赖来源。

之前 `Cute-Learning/common.mk` 会优先回退到：

- `../LLMQRT/runtime_refact/3rdparty/cutlass`
- `../LLMQRT_bak/runtime_refact/3rdparty/cutlass`

这会导致 `my_gemm` 虽然在 `Cute-Learning` 里开发，但实际却在吃别的仓库里的 CUTLASS。

现在已经改成：

- 统一只依赖 `flashdecoding/src/cutlass`

并且本地已经初始化过这个 submodule。

这意味着后续 `make info` 里看到的 `CUTLASS_ROOT` 应该是：

```text
/home/ai/workspace/Cute-Learning/flashdecoding/src/cutlass
```

当前 kernel 用的是：

```text
SM80_16x8x16_F16F16F16F16_TN
```

这表示当前 MMA 路径是：

- F16 输入
- F16 累加
- F16 输出

而当前 CPU 参考实现是：

- 用 float 做累加
- 最后再 cast 回 half

所以这两边并不是完全 apples-to-apples 的对比。
也就是说，当前 `0.125` 的误差更像是“参考标准更严格”，而不是“程序已经明显算错”。

## 我们接下来建议做什么

下一阶段建议按这个顺序推进：

1. 先统一 correctness 对比口径
   二选一：
   - 保持当前 GPU kernel 不变，把 CPU 参考实现改成 half accumulate
   - 保持当前 CPU 参考实现不变，把 GPU kernel 改成更高精度累加

2. 再确认当前 MMA / 输入布局 / CPU 参考实现是否完全一致。

3. 等 correctness 更稳后，再开始动 tile 或 shared memory 优化。

## 当前常用命令

```bash
cd /home/ai/workspace/Cute-Learning/my_gemm
make info
make build
make run
```

如果只想运行不重新编译：

```bash
./gemm
```

## 官方教程手写练习文件

这次额外新建了一个纯练习文件：

- `tutorial_gemm_scratch.cu`

它的用途不是直接提供实现，而是给你留一个干净入口，按 NVIDIA 官方 CuTe GEMM 教程自己一步一步手写。

当前这个文件里只保留了用途说明，没有放任何 kernel 实现。

如果你后面想单独编译它，不需要先改 `Makefile`，可以直接临时覆盖变量：

```bash
cd /home/ai/workspace/Cute-Learning/my_gemm
make build SRC=tutorial_gemm_scratch.cu TARGET=tutorial_gemm_scratch
```

如果你写完后想运行：

```bash
cd /home/ai/workspace/Cute-Learning/my_gemm
make run SRC=tutorial_gemm_scratch.cu TARGET=tutorial_gemm_scratch
```

## Nsight Systems 采样入口

这次给 `Makefile` 增加了一个 `nsys` target，用来生成 Nsight Systems 报告。

注意生成的文件后缀不是 `.nsy`，而是：

```text
.nsys-rep
```

当前 `Makefile` 里的默认设置是：

- 输出目录：`nsys_reports/`
- 输出前缀：`nsys_reports/my_gemm`
- trace 类型：`cuda,nvtx,osrt`

所以跑完后，主要报告文件会是：

```text
nsys_reports/my_gemm.nsys-rep
```

默认命令：

```bash
cd /home/ai/workspace/Cute-Learning/my_gemm
make nsys
```

如果你想换输出文件名，可以这样：

```bash
cd /home/ai/workspace/Cute-Learning/my_gemm
make nsys NSYS_OUT=nsys_reports/my_gemm_v1
```

如果你想清理这些 profiling 产物：

```bash
cd /home/ai/workspace/Cute-Learning/my_gemm
make nsys-clean
```

## 这次关于 nsys 的检查结论

本机已经有可用的 `nsys`：

- 路径：`/usr/local/cuda-13.1/bin/nsys`
- 版本：`2025.5.2.266-255236693005v0`

所以从工具 availability 来看，你现在可以直接尝试跑 profiling。

## 2026-04-19 调试记录

今天继续在老的 baseline 上做了 correctness 观察，当前运行输出如下：

```text
algo = my_first_cute_gemm
Mismatch at index 0: GPU = 62.812500, CPU = 62.781250, Diff = 0.031250
Mismatch at index 1: GPU = 68.187500, CPU = 68.250000, Diff = 0.062500
Mismatch at index 235: GPU = 67.875000, CPU = 67.750000, Diff = 0.125000
Correctness: M=256 N=256 K=256 Max Error = 0.125000
Performance: M=2048 N=2048 K=2048 Time = 0.001734 s, GFLOPS = 9225.78
```

这次输出说明了几件事：

1. 当前比较逻辑确实是逐元素比较 `h_out` 和 `h_ref` 的绝对误差。
2. 当前打印出来的是“部分 mismatch 样本”，不是把所有不一致元素都打出来。
3. 当前看到的最大误差点出现在 `index = 235`，对应 `Diff = 0.125000`。
4. 性能结果基本稳定在 `9.2 TFLOPS` 左右，说明 baseline 的运行状态是稳定的。

结合当前代码，比较链路是：

1. `cpu_f16_gemm_tn(h_a, h_b, h_ref, m, n, k);`
   先在 CPU 上算参考答案，结果写到 `h_ref`
2. `launch_my_first_gemm(d_a, d_b, d_c, m, n, k);`
   再在 GPU 上跑 CuTe kernel，结果写到 `d_c`
3. `cudaMemcpy(h_out, d_c, size_c, cudaMemcpyDeviceToHost);`
   把 GPU 结果拷回 `h_out`
4. `fabs(float(h_out[i]) - float(h_ref[i]))`
   做逐元素绝对误差比较，并更新 `max_error`

当前误差仍然更像“精度口径不一致”，而不是明显的功能性错误，主要原因还是：

- GPU 侧 MMA 路径是 `F16` 输入、`F16` 累加、`F16` 输出
- CPU 参考实现是 `float` 累加、最后再 cast 回 `half`

所以现在这份 baseline 更适合用来理解：

- CuTe 的 Tensor / tile / partition 语法
- kernel 的执行链路
- baseline 的正确性与性能趋势

而不适合直接把 `0.125` 当成“已经算错”的结论。
