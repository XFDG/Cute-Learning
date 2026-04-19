# CuTe 环境配置记录

## 先说结论

`Cute-Learning` 本身已经有一套可用的构建方式，不需要像完整 CUTLASS 仓库那样单独跑一遍大 CMake。

它当前的构建入口是：

- 根目录公共配置：`common.mk`
- 每个示例子目录自己的 `Makefile`

也就是说，你后面通常应该在 `Cute-Learning` 里这样做：

```bash
cd /home/ai/workspace/Cute-Learning/gemm/gemm_v1
make info
make build
make run
```

而不是去 `LLMQRT` 下面再单独配一套 CuTe 工程。

## 仓库结构里和 CuTe 最相关的部分

当前最值得关注的是这些目录：

- `gemm/gemm_v1`
- `gemm/gemm_v2`
- `gemm/gemm_v3`
- `gemm/gemm_v4`
- `gemv/gemv_cute`
- `others/tile_copy`
- `others/ldsm`

这些目录里的 `.cu` 示例都已经是基于 CuTe / CUTLASS 头文件来写的。

## Cute-Learning 当前是怎么找到 CUTLASS 的

关键逻辑在：

- `common.mk`

它会按顺序尝试这些候选路径：

1. `../LLMQRT/runtime_refact/3rdparty/cutlass`
2. `../LLMQRT_bak/runtime_refact/3rdparty/cutlass`
3. `flashdecoding/src/cutlass`
4. `sm90_decode/src/cutlass`

只要其中某个目录里存在：

```text
include/cute/tensor.hpp
```

它就会把那个目录当成 `CUTLASS_ROOT`。

在这台机器上，实际检测到的是：

```text
/home/ai/workspace/Cute-Learning/../LLMQRT/runtime_refact/3rdparty/cutlass
```

也就是说，`Cute-Learning` 目前是在复用 `LLMQRT` 里的 CUTLASS 头文件。

## 本机环境检查结果

本次检查到：

- 仓库：`/home/ai/workspace/Cute-Learning`
- GPU：`NVIDIA GeForce RTX 5060 Laptop GPU`
- Compute Capability：`12.0`
- 驱动：`591.74`
- `nvcc`：`/usr/local/cuda-13.1/bin/nvcc`
- CUDA 编译器版本：`13.1.115`

## Step 1: 先检查环境

我加了一个简单脚本：

- `scripts/check_cute_env.sh`

执行：

```bash
cd /home/ai/workspace/Cute-Learning
./scripts/check_cute_env.sh
```

它会输出：

- 当前用到的 `nvcc`
- GPU 和计算能力
- `gemm/gemm_v1` 里的 `make info`

## Step 2: 看看工程是否能自动识别 CUTLASS

你可以直接执行：

```bash
cd /home/ai/workspace/Cute-Learning/gemm/gemm_v1
make info
```

本次实际检查结果里，关键几项是：

- `DETECTED_GPU_ARCH=sm_120`
- `ARCH_FLAGS=-arch=sm_120`
- `CUTLASS_ROOT=/home/ai/workspace/Cute-Learning/../LLMQRT/runtime_refact/3rdparty/cutlass`

这说明：

1. GPU 架构自动识别是正常的。
2. 编译架构参数已经自动变成了 `sm_120`。
3. CUTLASS 头文件也已经被自动找到。

## Step 3: 编译一个最小 GEMM 示例

在 `gemm/gemm_v1` 目录执行：

```bash
make build
```

本次已经实测通过，说明 `Cute-Learning` 当前可以直接编译 CuTe GEMM 示例。

## Step 4: 运行验证

继续执行：

```bash
make run
```

本次已经成功跑起来，并输出了：

- `algo = Cute_HGEMM_V1`
- `Max Error = 0.187500`
- 后续一系列不同 `M/N/K` 的性能数据

这说明对 `Cute-Learning` 来说，环境已经不是主要阻塞点了。

## 现在还缺什么

如果你的目标是“在 Cute-Learning 里自己写一个 matmul”，现在真正还缺的是这些：

1. 一个你自己的实验目录
   最建议从 `gemm/gemm_v1` 复制一份开始。

2. 一份你自己的 `Makefile`
   直接复用现有模式即可：

```makefile
TARGET := gemm
SRC := gemm.cu
NEEDS_CUTLASS := 1

include ../../common.mk
```

3. 你自己的 kernel 目标
   也就是要决定你第一版到底做什么：
   - `half` 还是 `float`
   - Tensor Core 还是先做简单版本
   - 先追求正确性还是先追求性能

## 我建议你接下来的最稳路线

先按这个顺序来：

1. 先读 `gemm/gemm_v1/gemm.cu`
   它最接近“从零开始理解 CuTe matmul”。

2. 再读 `gemm/gemm_v2/gemm.cu`
   看它怎么开始做更完整的 tiled copy / tiled mma。

3. 再看 `gemm/gemm_v3` 和 `gemm/gemm_v4`
   这两版更偏优化。

4. 然后复制一份新的目录
   比如：

```text
gemm/gemm_my_first
```

第一版只做：

- 固定类型
- 固定 tile
- 固定布局
- 先保证结果对

## 当前仓库里可以直接用的命令

```bash
cd /home/ai/workspace/Cute-Learning
./scripts/check_cute_env.sh

cd /home/ai/workspace/Cute-Learning/gemm/gemm_v1
make info
make build
make run
```

## 下一步

下一步最合适的是直接在 `Cute-Learning` 里新建你自己的 matmul 目录，而不是继续在 `LLMQRT` 里折腾 CuTe 环境。

如果你要我继续，我下一步可以直接帮你做：

- 从 `gemm/gemm_v1` 复制出一个新的 `gemm/gemm_my_first`
- 改好 `Makefile`
- 给你留出一个最小可改的 CuTe matmul 模板
