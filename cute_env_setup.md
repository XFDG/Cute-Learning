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

现在已经统一固定为仓库内这一份：

1. `flashdecoding/src/cutlass`

它要求这个目录里存在：

```text
include/cute/tensor.hpp
```

如果这个目录还没初始化，应该先在仓库根目录执行：

```text
git submodule update --init --recursive flashdecoding/src/cutlass
```

也就是说，`Cute-Learning` 现在不再依赖 `LLMQRT` 或其他仓库里的 CUTLASS。

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
- `CUTLASS_ROOT=/home/ai/workspace/Cute-Learning/flashdecoding/src/cutlass`

这说明：

1. GPU 架构自动识别是正常的。
2. 编译架构参数已经自动变成了 `sm_120`。
3. 现在默认只会使用仓库内的 CUTLASS。

## Step 3: 编译一个最小 GEMM 示例

在 `gemm/gemm_v1` 目录执行：

```bash
make build
```

本次已经实测通过，说明 `Cute-Learning` 当前可以直接编译 CuTe GEMM 示例。

## 本次依赖修复记录

这次做了两类修复：

1. `common.mk`
   不再回退到：
   - `../LLMQRT/runtime_refact/3rdparty/cutlass`
   - `../LLMQRT_bak/runtime_refact/3rdparty/cutlass`

   而是统一固定到：

   - `flashdecoding/src/cutlass`

2. `sm90_decode/CMakeLists.txt`
   不再依赖它自己目录下那个空的 `src/cutlass`，
   而是统一引用 `../flashdecoding/src/cutlass`。

3. `flashdecoding/CMakeLists.txt`
   继续使用仓库内 `src/cutlass`，但现在会在 submodule 缺失时直接报出明确提示。

另外，本次已经实际初始化了仓库内的 CUTLASS submodule：

- `flashdecoding/src/cutlass`

所以现在 `my_gemm`、`gemm/gemm_v1` 这类依赖 `common.mk` 的目录，都会优先走仓库内 CUTLASS。

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
