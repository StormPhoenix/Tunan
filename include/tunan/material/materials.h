//
// Created by Storm Phoenix on 2021/6/6.
//

#ifndef TUNAN_MATERIALS_H
#define TUNAN_MATERIALS_H

#include <tunan/common.h>
#include <tunan/base/spectrum.h>
#include <tunan/base/interactions.h>
#include <tunan/material/bsdfs.h>
#include <tunan/material/Material.h>
#include <tunan/utils/MemoryAllocator.h>

namespace RENDER_NAMESPACE {
    namespace material {
        using utils::MemoryAllocator;
        using base::Spectrum;
        using base::SurfaceInteraction;
        using bsdf::LambertianBxDF;

        class Lambertian {
        public:
            using MaterialBxDF = LambertianBxDF;

            Lambertian(Spectrum Kd) : _Kd(Kd) {}

            RENDER_CPU_GPU BSDF evaluateBSDF(SurfaceInteraction &si, LambertianBxDF *bxdf,
                                             TransportMode mode = TransportMode::RADIANCE);

            RENDER_CPU_GPU bool isSpecular() {
                return false;
            }

        private:
            Spectrum _Kd;
        };

    }
}

#endif //TUNAN_MATERIALS_H
