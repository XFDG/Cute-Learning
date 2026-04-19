REPO_ROOT := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))

NVCC ?= nvcc
OPT_FLAGS ?= -O2
STD_FLAGS ?= -std=c++17
COMMON_NVCC_FLAGS ?= $(OPT_FLAGS) $(STD_FLAGS) --expt-relaxed-constexpr -cudart shared --cudadevrt none

DETECTED_GPU_ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '.')
ifeq ($(strip $(DETECTED_GPU_ARCH)),)
DETECTED_GPU_ARCH := 120
endif

GPU_ARCH ?= $(DETECTED_GPU_ARCH)
ARCH_FLAGS ?= -arch=sm_$(GPU_ARCH)

# Cute-Learning 统一只依赖仓库内这份 CUTLASS。
# 如果缺失，优先初始化 submodule，而不是回退到其他仓库里的 cutlass。
CUTLASS_ROOT := $(REPO_ROOT)/flashdecoding/src/cutlass

NEEDS_CUTLASS ?= 0
INCLUDE_FLAGS ?=
ifeq ($(NEEDS_CUTLASS),1)
ifeq ($(wildcard $(CUTLASS_ROOT)/include/cute/tensor.hpp),)
$(error Could not locate repo-local CUTLASS at $(CUTLASS_ROOT). Run 'git submodule update --init --recursive flashdecoding/src/cutlass' in the Cute-Learning repository)
endif
INCLUDE_FLAGS += -I$(CUTLASS_ROOT)/include
ifneq ($(wildcard $(CUTLASS_ROOT)/tools/util/include),)
INCLUDE_FLAGS += -I$(CUTLASS_ROOT)/tools/util/include
endif
endif

TARGET ?= app
SRC ?=
EXTRA_NVCC_FLAGS ?=
EXTRA_LIBS ?=
RUN_ARGS ?=
CUDA_DEVICE ?=
RUN_REQUIRES_ARCH ?=

RUN_PREFIX := $(if $(strip $(CUDA_DEVICE)),CUDA_VISIBLE_DEVICES=$(CUDA_DEVICE) ,)

.DEFAULT_GOAL := default

.PHONY: default build run clean info

default: build run

build:
	$(NVCC) -o $(TARGET) $(SRC) $(COMMON_NVCC_FLAGS) $(ARCH_FLAGS) $(INCLUDE_FLAGS) $(EXTRA_NVCC_FLAGS) $(EXTRA_LIBS)

run: build
	@if [ -n "$(RUN_REQUIRES_ARCH)" ] && [ "$(DETECTED_GPU_ARCH)" != "$(RUN_REQUIRES_ARCH)" ]; then \
		echo "Skipping ./$(TARGET): this example expects sm_$(RUN_REQUIRES_ARCH), detected sm_$(DETECTED_GPU_ARCH)."; \
		echo "You can still compile it, but running it on this GPU is not recommended."; \
	else \
		$(RUN_PREFIX)./$(TARGET) $(RUN_ARGS); \
	fi

clean:
	rm -f $(TARGET)

info:
	@echo "REPO_ROOT=$(REPO_ROOT)"
	@echo "TARGET=$(TARGET)"
	@echo "SRC=$(SRC)"
	@echo "DETECTED_GPU_ARCH=sm_$(DETECTED_GPU_ARCH)"
	@echo "GPU_ARCH=sm_$(GPU_ARCH)"
	@echo "ARCH_FLAGS=$(ARCH_FLAGS)"
	@echo "CUTLASS_ROOT=$(CUTLASS_ROOT)"
	@echo "INCLUDE_FLAGS=$(INCLUDE_FLAGS)"
	@echo "CUDA_DEVICE=$(CUDA_DEVICE)"
