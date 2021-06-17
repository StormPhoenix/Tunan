//
// Created by Storm Phoenix on 2021/6/10.
//

#ifndef TUNAN_TEXTURES_H
#define TUNAN_TEXTURES_H

#include <tunan/common.h>
#include <tunan/base/spectrum.h>
#include <tunan/base/interactions.h>
#include <tunan/material/mappings.h>
#include <tunan/utils/TaggedPointer.h>
#include <tunan/utils/ResourceManager.h>

namespace RENDER_NAMESPACE {
    namespace material {
        using base::Spectrum;
        using base::SurfaceInteraction;

        using utils::TaggedPointer;
        using utils::ResourceManager;

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

        class ChessboardSpectrumTexture {
        public:
            ChessboardSpectrumTexture(const Spectrum &color1, const Spectrum &color2, Float uScale, Float vScale);

            RENDER_CPU_GPU
            Spectrum evaluate(const SurfaceInteraction &si);

        private:
            Spectrum _color1;
            Spectrum _color2;
            Float _uScale;
            Float _vScale;
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

        class ChessboardFloatTexture {
        public:
            ChessboardFloatTexture(Float color1, Float color2, Float uScale, Float vScale);

            RENDER_CPU_GPU
            Float evaluate(const SurfaceInteraction &si);

        private:
            Float _color1;
            Float _color2;
            Float _uScale;
            Float _vScale;
        };

        class FloatTexture : public TaggedPointer<ConstantFloatTexture, ChessboardFloatTexture> {
        public:
            using TaggedPointer::TaggedPointer;

            RENDER_CPU_GPU
            Float evaluate(const SurfaceInteraction &si);
        };

        class SpectrumTexture : public TaggedPointer<ConstantSpectrumTexture, ChessboardSpectrumTexture> {
        public:
            using TaggedPointer::TaggedPointer;

            RENDER_CPU_GPU
            Spectrum evaluate(const SurfaceInteraction &si);
        };
    }
}

#endif //TUNAN_TEXTURES_H
