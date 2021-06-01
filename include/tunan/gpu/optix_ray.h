//
// Created by StormPhoenix on 2021/5/31.
//

#ifndef TUNAN_OPTIX_RAY_H
#define TUNAN_OPTIX_RAY_H

#include <optix.h>

struct RayParams {
    OptixTraversableHandle traversable;

    // TODO delete
    int width;
    int height;
    uchar3 *image;
};

// TODO for testing
struct RaygenData {
    float r = 0.5;
};

struct ClosestHitData {
    float r = 0.2;
};

#endif //TUNAN_OPTIX_RAY_H
