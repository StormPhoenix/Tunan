//
// Created by StormPhoenix on 2021/5/31.
//

#ifndef TUNAN_OPTIX_RAY_H
#define TUNAN_OPTIX_RAY_H

#include <optix.h>
#include <tunan/scene/Camera.h>

namespace RENDER_NAMESPACE {
    struct RayParams {
        Camera *camera;
        uchar3 *outputImage;
        OptixTraversableHandle traversable;
    };

// TODO for testing
    struct RayGenData {
        float r = 0.5;
    };

    struct ClosestHitData {
        float r = 0.2;
    };

    struct MissData {
        float b = 0.6;
    };
}

#endif //TUNAN_OPTIX_RAY_H
