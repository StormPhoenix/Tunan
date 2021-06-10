//
// Created by Storm Phoenix on 2021/6/6.
//

#ifndef TUNAN_MATERIALS_H
#define TUNAN_MATERIALS_H

#include <tunan/common.h>
#include <tunan/base/spectrum.h>
#include <tunan/base/interactions.h>
#include <tunan/material/bsdfs.h>
#include <tunan/utils/TaggedPointer.h>
#include <tunan/utils/MemoryAllocator.h>

namespace RENDER_NAMESPACE {
    namespace material {
        using utils::TaggedPointer;
        using utils::MemoryAllocator;

        using base::Spectrum;
        using base::SurfaceInteraction;

        using bsdf::BSDF;
        using bsdf::TransportMode;
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

        class Material : public TaggedPointer<Lambertian> {
        public:
            using TaggedPointer::TaggedPointer;

            RENDER_CPU_GPU inline bool isSpecular();
        };
    }
}

#endif //TUNAN_MATERIALS_H
