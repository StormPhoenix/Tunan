//
// Created by Graphics on 2021/5/31.
//

#include <tunan/material/Material.h>
#include <tunan/material/Lambertian.h>

namespace RENDER_NAMESPACE {
    namespace material {
        RENDER_CPU_GPU
        inline void Material::evaluateBSDF(SurfaceInteraction &insect, MemoryAllocator &allocator, TransportMode mode) {
            auto func = [&](auto ptr) { return ptr->evaluateBSDF(insect, allocator, mode); };
            return proxyCall(func);
        }

        RENDER_CPU_GPU
        inline bool Material::isSpecular() {
            auto func = [&](auto ptr) { return ptr->isSpecular(); };
            return proxyCall(func);
        }
    }
}