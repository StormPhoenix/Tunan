//
// Created by StormPhoenix on 2021/5/31.
//

#ifndef TUNAN_CUDAALLOCATOR_H
#define TUNAN_CUDAALLOCATOR_H

#include <tunan/common.h>
#include <tunan/utils/memory/Allocator.h>

namespace RENDER_NAMESPACE {
    namespace utils {
        class CudaAllocator : public Allocator {
        public:
            virtual void *allocateAlignedMemory(size_t bytes, size_t alignBytes) override;

            virtual void freeAlignedMemory(void *ptr) override;
        };
    }
}

#endif //TUNAN_CUDAALLOCATOR_H
