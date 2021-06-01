//
// Created by StormPhoenix on 2021/5/31.
//

#ifndef TUNAN_MATERIAL_H
#define TUNAN_MATERIAL_H

#include <tunan/common.h>
#include <tunan/base/interactions.h>
#include <tunan/bsdf/bxdf/BxDF.h>
#include <tunan/utils/TaggedPointer.h>
#include <tunan/utils/MemoryAllocator.h>

namespace RENDER_NAMESPACE {
    namespace material {
        using utils::TaggedPointer;
        using utils::MemoryAllocator;
        using base::SurfaceInteraction;
        using bsdf::TransportMode;

        class Lambertian;

        class Material : public TaggedPointer<Lambertian> {
        public:
            using TaggedPointer::TaggedPointer;

            RENDER_CPU_GPU inline void evaluateBSDF(SurfaceInteraction &insect, MemoryAllocator &allocator,
                                                    TransportMode mode = TransportMode::RADIANCE);

            RENDER_CPU_GPU inline bool isSpecular();
        };

    }

}
#endif //TUNAN_MATERIAL_H
