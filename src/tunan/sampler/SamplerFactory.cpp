//
// Created by StormPhoenix on 2021/6/5.
//

#include <tunan/sampler/SamplerFactory.h>

namespace RENDER_NAMESPACE {
    namespace sampler {
        Sampler SamplerFactory::newSampler(size_t nSamples, utils::ResourceManager *allocator) {
            return allocator->newObject<IndependentSampler>(nSamples);
        }
    }
}