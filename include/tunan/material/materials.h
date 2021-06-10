//
// Created by Storm Phoenix on 2021/6/6.
//

#ifndef TUNAN_MATERIALS_H
#define TUNAN_MATERIALS_H

#include <tunan/common.h>
#include <tunan/base/spectrum.h>
#include <tunan/base/interactions.h>
#include <tunan/material/bsdfs.h>
#include <tunan/material/textures.h>
#include <tunan/utils/TaggedPointer.h>
#include <tunan/utils/MemoryAllocator.h>

namespace RENDER_NAMESPACE {
    namespace material {
        using utils::TaggedPointer;
        using utils::MemoryAllocator;

        using base::Spectrum;
        using base::SurfaceInteraction;

        using namespace bsdf;

        class Lambertian {
        public:
            using MaterialBxDF = LambertianBxDF;

            Lambertian(SpectrumTexture Kd) : _Kd(Kd) {}

            RENDER_CPU_GPU BSDF evaluateBSDF(SurfaceInteraction &si, LambertianBxDF *bxdf,
                                             TransportMode mode = TransportMode::RADIANCE);

            RENDER_CPU_GPU inline bool isSpecular() {
                return false;
            }

        private:
            SpectrumTexture _Kd;
        };

        class Dielectric {
        public:
            using MaterialBxDF = FresnelSpecularBxDF;

            Dielectric(SpectrumTexture R, SpectrumTexture T, Float etaI, Float etaT, Float roughness = 0.f);

            RENDER_CPU_GPU bool isSpecular() const;

            RENDER_CPU_GPU BSDF evaluateBSDF(SurfaceInteraction &si, FresnelSpecularBxDF *bxdf,
                                             TransportMode mode = TransportMode::RADIANCE);

        private:
            Float _roughness;
            Float _etaI, _etaT;
            SpectrumTexture _R;
            SpectrumTexture _T;
        };


        class Material : public TaggedPointer<Lambertian, Dielectric> {
        public:
            using TaggedPointer::TaggedPointer;

            RENDER_CPU_GPU inline bool isSpecular();
        };
    }
}

#endif //TUNAN_MATERIALS_H
