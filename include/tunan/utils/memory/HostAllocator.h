//
// Created by StormPhoenix on 2021/5/31.
//

#ifndef TUNAN_HOSTALLOCATOR_H
#define TUNAN_HOSTALLOCATOR_H

#include <tunan/common.h>
#include <tunan/utils/memory/Allocator.h>

namespace RENDER_NAMESPACE {
    namespace utils {
        class HostAllocator : public Allocator {
        public:
            virtual void *allocateAlignedMemory(size_t bytes, size_t alignBytes) override;

            virtual void freeAlignedMemory(void *ptr) override;
        };
    }
}

#endif //TUNAN_HOSTALLOCATOR_H
