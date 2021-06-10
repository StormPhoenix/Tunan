//
// Created by Storm Phoenix on 2021/6/10.
//

#include <tunan/material/textures.h>

namespace RENDER_NAMESPACE {
    namespace material {
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
    }
}