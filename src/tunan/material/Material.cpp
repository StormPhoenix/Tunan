//
// Created by StormPhoenix on 2021/5/31.
//

#include <tunan/material/Material.h>
#include <tunan/material/materials.h>

namespace RENDER_NAMESPACE {
    namespace material {

        RENDER_CPU_GPU
        inline bool Material::isSpecular() {
            auto func = [&](auto ptr) { return ptr->isSpecular(); };
            return proxyCall(func);
        }
    }
}