//
// Created by StormPhoenix on 2021/5/31.
//

#ifndef TUNAN_INTERACTIONS_H
#define TUNAN_INTERACTIONS_H

#include <tunan/math.h>
#include <tunan/common.h>
#include <tunan/base/mediums.h>
#include <tunan/scene/Ray.h>

namespace RENDER_NAMESPACE {
    namespace base {
        class Interaction {
        public:
            RENDER_CPU_GPU
            Interaction() = default;

            RENDER_CPU_GPU
            Interaction(Point3F p, Normal3F ng, Vector3F wo, Point2F uv)
                    : p(p), ng(ng), uv(uv), wo(NORMALIZE(wo)) {}

            Point3F p;
            Vector3F wo;
            Normal3F ng;
            Point2F uv;
            const MediumInterface *mediumInterface = nullptr;
        };

        class SurfaceInteraction : public Interaction {
        public:
            RENDER_CPU_GPU
            SurfaceInteraction() = default;

            RENDER_CPU_GPU
            SurfaceInteraction(Point3F p, Normal3F ng, Normal3F ns, Vector3F wo, Point2F st, Vector3F error)
                    : Interaction(p, ng, wo, st), ns(ns), error(error) {}

            RENDER_CPU_GPU
            Ray generateRay(const Vector3F &direction) const;

            Normal3F ns;
            Vector3F error;
        };
    }
}
#endif //TUNAN_INTERACTIONS_H
