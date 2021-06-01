//
// Created by StormPhoenix on 2021/5/31.
//

#ifndef TUNAN_CPURESOURCE_H
#define TUNAN_CPURESOURCE_H

#include <tunan/common.h>
#include <tunan/utils/memory/MemoryResource.h>

namespace RENDER_NAMESPACE {
    namespace utils {
        class CPUResource : public MemoryResource {
        public:
            virtual void *allocateAlignedMemory(size_t bytes, size_t alignBytes) override;

            virtual void freeAlignedMemory(void *ptr) override;
        };
    }
}

#endif //TUNAN_CPURESOURCE_H
