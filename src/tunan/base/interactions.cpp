//
// Created by StormPhoenix on 2021/5/31.
//

#include <tunan/common.h>
#include <tunan/base/interactions.h>

namespace RENDER_NAMESPACE {
    namespace base {
        RENDER_CPU_GPU
        Vector3F offsetOrigin(const Vector3F &origin, const Vector3F &error,
                              const Vector3F &normal, const Vector3F &direction) {
            Float dist = 2.0 * DOT(ABS(normal), error);
            Vector3F offset = dist * normal;
            if (DOT(normal, direction) < 0) {
                offset = -offset;
            }

            Point3F o = origin + offset;
            for (int i = 0; i < 3; i++) {
                if (offset[i] > 0) {
                    o[i] = math::floatUp(o[i]);
                } else if (offset[i] < 0) {
                    o[i] = math::floatDown(o[i]);
                }
            }
            return o;
        }

        RENDER_CPU_GPU
        Ray SurfaceInteraction::generateRay(const Vector3F &direction) const {
            Vector3F origin = offsetOrigin(p, error, ng, direction);
            return Ray(origin, NORMALIZE(direction));
        }

        RENDER_CPU_GPU
        Ray SurfaceInteraction::generateRayTo(const Interaction &target) const {
            // check whether the ray direction is point to outside or inside
            Vector3F pOrigin = offsetOrigin(p, error, ng, target.p - p);
            Vector3F pTarget = offsetOrigin(target.p, target.error, target.ng, p - target.p);
            const Vector3F direction = (pTarget - pOrigin);
            Float step = LENGTH(direction);

            // TODO medium
            Ray ray = Ray(pOrigin, NORMALIZE(direction));
            ray.setStep(step - ShadowEpsilon);
            return ray;
        }
    }
}