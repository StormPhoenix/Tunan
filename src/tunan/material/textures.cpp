//
// Created by Storm Phoenix on 2021/6/10.
//

#include <tunan/material/textures.h>

namespace RENDER_NAMESPACE {
    namespace material {
        RENDER_CPU_GPU
        Spectrum ImageSpectrumTexture::evaluate(const SurfaceInteraction &si) {
            if (_texture == nullptr) {
                return Spectrum(0);
            }
            Point2F uv = _textureMapping.map(si);

            int wOffset, hOffset;
            wOffset = uv[0] * _width;
            hOffset = uv[1] * _height;

            if (wOffset < 0 || wOffset >= _width
                || hOffset < 0 || hOffset >= _height) {
                return Spectrum(0);
            }

            // flip
            int offset = (hOffset * _width + wOffset);

            Spectrum ret(0);
            for (int ch = 0; ch < _channel && ch < SpectrumChannel; ch++) {
                ret[ch] = Float(_texture[offset][ch]);
            }
            return ret;
        }

        RENDER_CPU_GPU
        Spectrum SpectrumTexture::evaluate(const SurfaceInteraction &si) {
            auto func = [&](auto ptr) {
                return ptr->evaluate(si);
            };
            return proxyCall(func);
        }

        RENDER_CPU_GPU
        Float FloatTexture::evaluate(const SurfaceInteraction &si) {
            auto func = [&](auto ptr) {
                return ptr->evaluate(si);
            };
            return proxyCall(func);
        }

        ChessboardSpectrumTexture::ChessboardSpectrumTexture(const Spectrum &color1, const Spectrum &color2,
                                                             Float uScale, Float vScale)
                : _color1(color1), _color2(color2), _uScale(uScale), _vScale(vScale) {}

        RENDER_CPU_GPU
        Spectrum ChessboardSpectrumTexture::evaluate(const SurfaceInteraction &si) {
            Point2F uv = Point2F(si.uv[0] * _uScale, si.uv[1] * _vScale);
            if ((int(uv[0]) + int(uv[1])) % 2 == 0) {
                return _color1;
            } else {
                return _color2;
            }
        }

        ChessboardFloatTexture::ChessboardFloatTexture(Float color1, Float color2, Float uScale, Float vScale)
                : _color1(color1), _color2(color2), _uScale(uScale), _vScale(vScale) {}

        RENDER_CPU_GPU
        Float ChessboardFloatTexture::evaluate(const SurfaceInteraction &si) {
            Point2F uv = Point2F(si.uv[0] * _uScale, si.uv[1] * _vScale);
            if ((int(uv[0]) + int(uv[1])) % 2 == 0) {
                return _color1;
            } else {
                return _color2;
            }
        }
    }
}