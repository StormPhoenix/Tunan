//
// Created by StormPhoenix on 2021/6/5.
//

#ifndef TUNAN_SAMPLERFACTORY_H
#define TUNAN_SAMPLERFACTORY_H

#include <tunan/common.h>
#include <tunan/sampler/samplers.h>
#include <tunan/utils/MemoryAllocator.h>

namespace RENDER_NAMESPACE {
    namespace sampler {
        class SamplerFactory {
        public:
            static  Sampler newSampler(size_t nSamples, utils::MemoryAllocator &allocator);
        };
    }
}

#endif //TUNAN_SAMPLERFACTORY_H
