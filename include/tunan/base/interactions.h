//
// Created by StormPhoenix on 2021/5/31.
//

#ifndef TUNAN_INTERACTIONS_H
#define TUNAN_INTERACTIONS_H

#include <tunan/math.h>
#include <tunan/common.h>
#include <tunan/scene/ray.h>
#include <tunan/medium/mediums.h>
#include <tunan/medium/phase_functions.h>

namespace RENDER_NAMESPACE {
    namespace base {
        class Interaction {
        public:
            RENDER_CPU_GPU
            Interaction() = default;

            RENDER_CPU_GPU
            Interaction(Point3F p, Normal3F ng, Vector3F wo)
                    : p(p), ng(ng), wo(NORMALIZE(wo)), error(0) {}

            RENDER_CPU_GPU
            Interaction(Point3F p, Normal3F ng, Vector3F wo, Vector3F error)
                    : p(p), ng(ng), wo(NORMALIZE(wo)), error(error) {}

            RENDER_CPU_GPU
            Interaction(Point3F p, Normal3F ng, Vector3F wo, Vector3F error, MediumInterface mediumInterface)
                    : p(p), ng(ng), wo(NORMALIZE(wo)), error(error), mediumInterface(mediumInterface) {}

            Point3F p;
            Vector3F wo;
            Normal3F ng;
            Vector3F error;
            MediumInterface mediumInterface;
        };

        class SurfaceInteraction : public Interaction {
        public:
            RENDER_CPU_GPU
            SurfaceInteraction() = default;

            RENDER_CPU_GPU
            SurfaceInteraction(Point3F p, Normal3F ng, Normal3F ns, Vector3F wo, Point2F st, Vector3F error)
                    : Interaction(p, ng, wo, error), ns(ns), uv(st) {}

            RENDER_CPU_GPU
            SurfaceInteraction(Point3F p, Normal3F ng, Normal3F ns, Vector3F wo, Point2F st, Vector3F error,
                               MediumInterface mediumInterface)
                    : Interaction(p, ng, wo, error, mediumInterface), ns(ns), uv(st) {}

            RENDER_CPU_GPU
            Ray generateRay(const Vector3F &direction) const;

            RENDER_CPU_GPU
            Ray generateRayTo(const Interaction &target) const;

            Point2F uv;
            Normal3F ns;
        };

        class MediumInteraction : public Interaction {
        public:
            RENDER_CPU_GPU
            MediumInteraction() = default;

            RENDER_CPU_GPU
            MediumInteraction(Point3F p, Vector3F wo, Medium medium, PhaseFunction phaseFunc)
                    : Interaction(p, wo, wo, Vector3F(0), MediumInterface(medium, medium)),
                      medium(medium), phaseFunc(phaseFunc) {}

            Medium medium;
            PhaseFunction phaseFunc;
        };
    }
}
#endif //TUNAN_INTERACTIONS_H
