#include <optix.h>

extern "C" __global__ void __raygen__findIntersection() {
    uint3 launch_index = optixGetLaunchIndex();
}