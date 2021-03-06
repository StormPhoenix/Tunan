//
// Created by StormPhoenix on 2021/5/31.
//

#ifndef TUNAN_OPTIX_RAY_H
#define TUNAN_OPTIX_RAY_H

#include <optix.h>
#include <tunan/tracer/tracer.h>
#include <tunan/scene/meshes.h>
#include <tunan/scene/cameras.h>
#include <tunan/scene/lights.h>
#include <tunan/material/materials.h>
#include <tunan/medium/mediums.h>

namespace RENDER_NAMESPACE {
    using namespace tracer;
    using namespace material;

    struct RayParams {
        RayQueue *rayQueue;
        MissQueue *missQueue;
        MediaEvaQueue *mediaEvaQueue;
        MaterialEvaQueue *materialEvaQueue;
        AreaLightHitQueue *areaLightQueue;
        PixelStateArray *pixelStateArray;
        ShadowRayQueue *shadowRayQueue;
        OptixTraversableHandle traversable;
    };

    struct RayGenData {
        // TODO delete
        float r = 0.5;
    };

    struct ClosestHitData {
        TriangleMesh *mesh = nullptr;
        DiffuseAreaLight *areaLights = nullptr;
        Material material;
        MediumInterface mediumInterface;
    };

    struct MissData {
        // TODO delete
        float b = 0.6;
    };
}

#endif //TUNAN_OPTIX_RAY_H
