//
// Created by StormPhoenix on 2021/5/31.
//

#ifndef TUNAN_ALLOCATOR_H
#define TUNAN_ALLOCATOR_H

#include <tunan/common.h>
#include <stddef.h>

namespace RENDER_NAMESPACE {
    namespace utils {
        class Allocator {
        public:
            virtual void *allocateAlignedMemory(size_t bytes, size_t alignBytes) = 0;

            virtual void freeAlignedMemory(void *ptr) = 0;
        };
    }
}

#endif //TUNAN_ALLOCATOR_H
