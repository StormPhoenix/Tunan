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
#include <tunan/utils/image_utils.h>

#include <sstream>
#include <istream>
#include <string>
#include <fstream>

namespace RENDER_NAMESPACE {
    namespace material {
        using namespace base;
        using namespace utils;

        class ImageSpectrumTexture {
        public:
            ImageSpectrumTexture() {}

            ImageSpectrumTexture(std::string imagePath, TextureMapping2D textureMapping, ResourceManager *allocator) {

                /* Check file exists */
                {
                    std::ifstream in(imagePath);
                    if (!in.good()) {
                        std::cout << imagePath << " NOT EXISTS." << std::endl;
                        return;
                    }
                }

                // Mapping
                {
                    if (!textureMapping.nullable()) {
                        _textureMapping = textureMapping;
                    } else {
                        _textureMapping = allocator->newObject<UVMapping2D>();
                    }
                }

                // Copy image to texture
                int channelsInFile = 3;
                _channel = 3;
                {
                    float *image = readImage(imagePath, &_width, &_height, _channel, &channelsInFile);
                    _texture = allocator->allocateObjects<Spectrum>(_width * _height);
                    for (int row = 0; row < _height; row++) {
                        for (int col = 0; col < _width; col++) {
                            int imageOffset = (row * _width + col) * channelsInFile;
                            int textureOffset = (row * _width + col);

                            for (int ch = 0; ch < _channel; ch++) {
                                (*(_texture + textureOffset))[ch] = *(image + imageOffset + ch);
                            }
                        }
                    }
                    delete image;
                }

                if (_channel > SpectrumChannel) {
                    std::cout << "Image channel count NOT Support !" << std::endl;
                }
            }

            RENDER_CPU_GPU
            Spectrum evaluate(const SurfaceInteraction &si);

        private:
            TextureMapping2D _textureMapping;
            Spectrum *_texture = nullptr;
            int _width, _height;
            int _channel;
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

        class SpectrumTexture : public TaggedPointer<ConstantSpectrumTexture, ChessboardSpectrumTexture,
                ImageSpectrumTexture> {
        public:
            using TaggedPointer::TaggedPointer;

            RENDER_CPU_GPU
            Spectrum evaluate(const SurfaceInteraction &si);
        };
    }
}

#endif //TUNAN_TEXTURES_H
