//
// Created by StormPhoenix on 2021/6/5.
//

#ifndef TUNAN_PARALLELS_H
#define TUNAN_PARALLELS_H

#include <tunan/common.h>
#include <tunan/base/containers.h>

#ifdef __RENDER_GPU_MODE__

#include <cuda_runtime.h>
#include <tunan/gpu/cuda_utils.h>

#endif

#include <map>
#include <typeindex>

namespace RENDER_NAMESPACE {
    namespace parallel {

#ifdef __RENDER_GPU_MODE__

        template<typename F>
        __global__ void kernel1D(F func, int count) {
            int threadId = blockIdx.x * blockDim.x + threadIdx.x;
            if (threadId >= count) {
                return;
            }
            func(threadId);
        }

        template<typename F>
        static void gpuParallelFor(F func, int count) {
            static std::map<std::type_index, int> typeBlockSizeMap;

            auto kernel = &kernel1D<F>;
            int blockSize;
            std::type_index typeIndex = std::type_index(typeid(F));
            auto iter = typeBlockSizeMap.find(typeIndex);
            if (iter != typeBlockSizeMap.end()) {
                blockSize = iter->second;
            } else {
                int minGridSize;
                CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel, 0, 0));
                typeBlockSizeMap[typeIndex] = blockSize;
            }

            int gridSize = (count + blockSize - 1) / blockSize;
            kernel<<<gridSize, blockSize>>>(func, count);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

#endif

        template<typename F>
        void parallelFor(F func, int count) {
#ifdef __RENDER_GPU_MODE__
            gpuParallelFor(func, count);
#endif
            // TODO unimplemented
        }

        template<typename F, typename QueueItem>
        void parallelForQueue(F func, const base::Queue <QueueItem> *queue, size_t maxQueueSize) {
#ifdef __RENDER_GPU_MODE__
            auto f = [=] RENDER_GPU(int idx) mutable {
                if (idx >= queue->size()) {
                    return;
                }
                func((*queue)[idx]);
            };
            gpuParallelFor(f, maxQueueSize);
#endif
            // TODO unimplemented
        }
    }
}
#endif //TUNAN_PARALLELS_H
