//
// Created by StormPhoenix on 2021/5/31.
//

#ifndef TUNAN_OPTIX_RAY_H
#define TUNAN_OPTIX_RAY_H

#include <optix.h>
#include <tunan/tracer/tracer.h>
#include <tunan/scene/TriangleMesh.h>
#include <tunan/scene/Camera.h>

namespace RENDER_NAMESPACE {
    using namespace tracer;

    struct RayParams {
        RayQueue *rayQueue;
        uchar3 *outputImage;
        OptixTraversableHandle traversable;
    };

    struct RayGenData {
        float r = 0.5;
    };

    struct ClosestHitData {
        TriangleMesh *mesh;
    };

    struct MissData {
        float b = 0.6;
    };
}

#endif //TUNAN_OPTIX_RAY_H
