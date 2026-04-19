#pragma once

#include <cstdio>
#include <cstdlib>

#define CHECK_CUDA(expr)                                                      \
  do {                                                                        \
    cudaError_t err__ = (expr);                                               \
    if (err__ != cudaSuccess) {                                               \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                   cudaGetErrorString(err__));                                \
      std::exit(EXIT_FAILURE);                                                \
    }                                                                         \
  } while (0)

#define PRINT(name, content) \
  print(name);               \
  print(" : ");              \
  print(content);            \
  print("\n");

#define PRINTTENSOR(name, content) \
  print(name);                     \
  print(" : ");                    \
  print_tensor(content);           \
  print("\n");
