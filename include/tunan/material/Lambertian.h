//
// Created by StormPhoenix on 2021/5/31.
//

#ifndef TUNAN_LAMBERTIAN_H
#define TUNAN_LAMBERTIAN_H

#include <tunan/common.h>
#include <tunan/bsdf/bxdf/BxDF.h>
#include <tunan/base/spectrum.h>
#include <tunan/base/interactions.h>
#include <tunan/utils/MemoryAllocator.h>

namespace RENDER_NAMESPACE {
    namespace material {
        using utils::MemoryAllocator;
        using base::Spectrum;
        using base::SurfaceInteraction;
        using bsdf::TransportMode;

        class Lambertian {
        public:
            Lambertian(Spectrum Kd) : _Kd(Kd) {}

            RENDER_CPU_GPU void evaluateBSDF(SurfaceInteraction &si, MemoryAllocator &allocator,
                                             TransportMode mode = TransportMode::RADIANCE);

            RENDER_CPU_GPU bool isSpecular() {
                return false;
            }

        private:
            Spectrum _Kd;
        };
    }
}

#endif //TUNAN_LAMBERTIAN_H
