//
// Created by StormPhoenix on 2021/5/31.
//

#ifndef TUNAN_CUDA_UTILS_H
#define TUNAN_CUDA_UTILS_H

#include <tunan/common.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#ifdef __RENDER_GPU_MODE__

#include <string>
#include <iostream>

#define CUDA_CHECK(exp) \
    if (exp != cudaSuccess) { \
        cudaError_t error = cudaGetLastError(); \
        ASSERT(false, "CUDA error: " + std::string(cudaGetErrorString(error))); \
    } else {}

#define CU_CHECK(exp)                                              \
    do {                                                            \
        CUresult result = exp;                                     \
        if (result != CUDA_SUCCESS) {                               \
            const char *str;                                        \
            assert(CUDA_SUCCESS == cuGetErrorString(result, &str));\
            ASSERT(false, "CUDA error: " + std::string(str));       \
        }                                                           \
    } while (false)

#else

#define CUDA_CHECK(exp) ASSERT(false, "CUDA error: unimplemented");

#endif

#endif //TUNAN_CUDA_UTILS_H
