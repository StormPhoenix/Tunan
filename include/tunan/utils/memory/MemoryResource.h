//
// Created by StormPhoenix on 2021/5/31.
//

#ifndef TUNAN_MEMORYRESOURCE_H
#define TUNAN_MEMORYRESOURCE_H

#include <tunan/common.h>

namespace RENDER_NAMESPACE {
    namespace utils {
        class MemoryResource {
        public:
            virtual void *allocateAlignedMemory(size_t bytes, size_t alignBytes) = 0;

            virtual void freeAlignedMemory(void *ptr) = 0;
        };
    }
}

#endif //TUNAN_MEMORYRESOURCE_H
