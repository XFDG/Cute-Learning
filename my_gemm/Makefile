TARGET := gemm
SRC := gemm.cu
NEEDS_CUTLASS := 1
NSYS ?= nsys
NSYS_DIR ?= nsys_reports
NSYS_OUT ?= $(NSYS_DIR)/my_gemm
NSYS_FLAGS ?= --trace=cuda,nvtx,osrt --sample=none --force-overwrite=true

.PHONY: nsys nsys-clean

nsys: build
	mkdir -p $(NSYS_DIR)
	$(RUN_PREFIX)$(NSYS) profile $(NSYS_FLAGS) -o $(NSYS_OUT) ./$(TARGET) $(RUN_ARGS)
	@echo "Generated report: $(NSYS_OUT).nsys-rep"

nsys-clean:
	rm -f $(NSYS_OUT).nsys-rep $(NSYS_OUT).sqlite $(NSYS_OUT).qdstrm

include ../common.mk
