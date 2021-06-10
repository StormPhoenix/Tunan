//
// Created by Storm Phoenix on 2021/6/6.
//

#include <tunan/material/materials.h>
#include <tunan/material/bsdfs.h>

namespace RENDER_NAMESPACE {
    namespace material {
        using bsdf::LambertianBxDF;

        RENDER_CPU_GPU
        BSDF Lambertian::evaluateBSDF(SurfaceInteraction &si, LambertianBxDF *bxdf, TransportMode mode) {
            // TODO delete
//            Spectrum albedo = _Kd->evaluate(insect);
            (*bxdf) = LambertianBxDF(_Kd);
            BSDF bsdf = BSDF(si.ng, si.ns, si.wo);
            bsdf.setBxDF(bxdf);
            return bsdf;
        }


        RENDER_CPU_GPU
        inline bool Material::isSpecular() {
            auto func = [&](auto ptr) { return ptr->isSpecular(); };
            return proxyCall(func);
        }
    }
}