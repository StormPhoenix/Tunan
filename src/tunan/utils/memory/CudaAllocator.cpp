//
// Created by StormPhoenix on 2021/5/31.
//

#include <tunan/utils/memory/CudaAllocator.h>
#include <tunan/gpu/cuda_utils.h>

#include <cuda.h>
#include <cuda_runtime.h>

namespace RENDER_NAMESPACE {
    namespace utils {
        void *CudaAllocator::allocateAlignedMemory(size_t bytes, size_t alignBytes) {
            void *ptr;
            CUDA_CHECK(cudaMallocManaged(&ptr, bytes));
            return ptr;
        }

        void CudaAllocator::freeAlignedMemory(void *ptr) {
            CUDA_CHECK(cudaFree(ptr));
        }
    }
}