//
// Created by Storm Phoenix on 2021/6/10.
//

#include <tunan/material/mappings.h>

namespace RENDER_NAMESPACE {
    namespace material {
        RENDER_CPU_GPU
        Point2F TextureMapping2D::map(const SurfaceInteraction &si) {
            auto func = [&] (auto ptr) {
                return ptr->map(si);
            };
            return proxyCall(func);
        }
    }
}