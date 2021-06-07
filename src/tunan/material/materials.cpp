//
// Created by Storm Phoenix on 2021/6/6.
//

#include <tunan/material/materials.h>
#include <tunan/material/bsdfs.h>

namespace RENDER_NAMESPACE {
    namespace material {
        using bsdf::LambertianBxDF;

        RENDER_CPU_GPU
        BSDF Lambertian::evaluateBSDF(SurfaceInteraction &si, MemoryAllocator &allocator, TransportMode mode) {
            // TODO delete
//            Spectrum albedo = _Kd->evaluate(insect);

//            LambertianBxDF bxdf = allocator.newObject<LambertianBxDF>(_Kd);
//            insect.bsdf = allocator.newObject<BSDF>(insect);
//            insect.bsdf->addBXDF(lambertianBXDF);
        }
    }
}