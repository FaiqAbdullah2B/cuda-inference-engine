#ifndef ENGINE_CUDA_UTILS_H
#define ENGINE_CUDA_UTILS_H

#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <float.h>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N)) // macro for ceil division

void cuda_check(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
               cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
};
#define cudaCheck(err) (cuda_check(err, __FILE__, __LINE__))

#endif