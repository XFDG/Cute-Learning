# my_gemm 开发记录

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
