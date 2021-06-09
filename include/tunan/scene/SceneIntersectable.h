//
// Created by StormPhoenix on 2021/6/6.
//

#ifndef TUNAN_SCENEINTERSECTABLE_H
#define TUNAN_SCENEINTERSECTABLE_H

#include <tunan/common.h>
#include <tunan/tracer/tracer.h>

namespace RENDER_NAMESPACE {
    using namespace tracer;

    class SceneIntersectable {
    public:
        virtual void findClosestHit(RayQueue *rayQueue, MissQueue *missQueue, MaterialEvaQueue *materialEvaQueue,
                               MediaEvaQueue *mediaEvaQueue, AreaLightHitQueue *areaLightQueue,
                               PixelStateArray *pixelStateArray) = 0;

        virtual void traceShadowRay(ShadowRayQueue *shadowRayQueue, PixelStateArray *pixelStateArray) = 0;
    };
}

#endif //TUNAN_SCENEINTERSECTABLE_H
