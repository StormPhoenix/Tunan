//
// Created by Graphics on 2021/5/31.
//

#include <tunan/utils/memory/CUDAResource.h>
#include <tunan/gpu/cuda_utils.h>

#include <cuda.h>
#include <cuda_runtime.h>

namespace RENDER_NAMESPACE {
    namespace utils {
        void *CUDAResource::allocateAlignedMemory(size_t bytes, size_t alignBytes) {
            void *ptr;
            CUDA_CHECK(cudaMallocManaged(&ptr, bytes));
            return ptr;
        }

        void CUDAResource::freeAlignedMemory(void *ptr) {
            CUDA_CHECK(cudaFree(ptr));
        }
    }
}