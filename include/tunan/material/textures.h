//
// Created by Storm Phoenix on 2021/6/10.
//

#ifndef TUNAN_TEXTURES_H
#define TUNAN_TEXTURES_H

#include <tunan/common.h>
#include <tunan/base/spectrum.h>
#include <tunan/base/interactions.h>
#include <tunan/utils/TaggedPointer.h>
#include <tunan/utils/MemoryAllocator.h>

namespace RENDER_NAMESPACE {
    namespace material {
        using base::Spectrum;
        using base::SurfaceInteraction;

        using utils::TaggedPointer;
        using utils::MemoryAllocator;

        class ImageSpectrumTexture {
        };

        class ConstantSpectrumTexture {
        public:
            ConstantSpectrumTexture() {
                _value = Spectrum(0.f);
            }

            ConstantSpectrumTexture(const Spectrum &value) : _value(value) {}

            RENDER_CPU_GPU
            Spectrum evaluate(const SurfaceInteraction &si) {
                return _value;
            }

        private:
            Spectrum _value;
        };

        class ConstantFloatTexture {
        public:
            ConstantFloatTexture() {
                _value = 0.f;
            }

            ConstantFloatTexture(Float value) : _value(value) {}

            RENDER_CPU_GPU
            Float evaluate(const SurfaceInteraction &si) {
                return _value;
            }

        private:
            Float _value;
        };

        class FloatTexture : public TaggedPointer<ConstantFloatTexture> {
        public:
            RENDER_CPU_GPU
            Float evaluate(const SurfaceInteraction &si);
        };

        class SpectrumTexture : public TaggedPointer<ConstantSpectrumTexture> {
        public:
            RENDER_CPU_GPU
            Spectrum evaluate(const SurfaceInteraction &si);
        };
    }
}

#endif //TUNAN_TEXTURES_H
