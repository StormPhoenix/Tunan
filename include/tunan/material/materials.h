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
#include <tunan/utils/ResourceManager.h>

namespace RENDER_NAMESPACE {
    namespace material {
        using namespace base;
        using namespace bsdf;
        using namespace utils;

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

        class Mirror {
        public:
            using MaterialBxDF = SpecularReflectionBxDF;

            Mirror();

            Mirror(SpectrumTexture &Ks);

            RENDER_CPU_GPU
            inline bool isSpecular() const {
                return true;
            }

            RENDER_CPU_GPU
            BSDF evaluateBSDF(SurfaceInteraction &si, SpecularReflectionBxDF *bxdf,
                              TransportMode mode = TransportMode::RADIANCE);

        private:
            SpectrumTexture _Ks;
        };

        class Metal {
        public:
            using MaterialBxDF = ConductorBxDF;

            Metal(FloatTexture alpha, SpectrumTexture eta, SpectrumTexture Ks, SpectrumTexture K,
                  MicrofacetDistribType distribType);

            RENDER_CPU_GPU BSDF evaluateBSDF(SurfaceInteraction &si, ConductorBxDF *bxdf,
                                             TransportMode mode = TransportMode::RADIANCE);

            RENDER_CPU_GPU inline bool isSpecular() const {
                return false;
            }

        private:
            FloatTexture _alpha;
            SpectrumTexture _eta;
            SpectrumTexture _Ks;
            SpectrumTexture _K;
            MicrofacetDistribType _distribType;
        };


        class Material : public TaggedPointer<Lambertian, Dielectric, Mirror, Metal> {
        public:
            using TaggedPointer::TaggedPointer;

            RENDER_CPU_GPU inline bool isSpecular();
        };
    }
}

#endif //TUNAN_MATERIALS_H
