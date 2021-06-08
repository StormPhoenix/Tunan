//
// Created by StormPhoenix on 2021/5/31.
//

#ifndef TUNAN_OPTIX_RAY_H
#define TUNAN_OPTIX_RAY_H

#include <optix.h>
#include <tunan/tracer/tracer.h>
#include <tunan/scene/TriangleMesh.h>
#include <tunan/scene/Camera.h>
#include <tunan/scene/lights.h>
#include <tunan/material/Material.h>

namespace RENDER_NAMESPACE {
    using namespace tracer;
    using namespace material;

    struct RayParams {
        RayQueue *rayQueue;
        MissQueue *missQueue;
        MaterialEvaQueue *materialEvaQueue;
        PixelStateArray *pixelStateArray;
        uchar3 *outputImage;
        OptixTraversableHandle traversable;
    };

    struct RayGenData {
        float r = 0.5;
    };

    struct ClosestHitData {
        TriangleMesh *mesh = nullptr;
        Material material;
        DiffuseAreaLight *areaLights = nullptr;
    };

    struct MissData {
        float b = 0.6;
    };
}

#endif //TUNAN_OPTIX_RAY_H
