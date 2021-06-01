#include <optix.h>
#include <tunan/gpu/optix_ray.h>

extern "C" {
__constant__ RayParams params;
}

extern "C" __global__ void __raygen__findIntersection() {
    uint3 launch_index = optixGetLaunchIndex();

    // TODO delete
    uchar3 t;
    t.x = 0;
    t.y = 200;
    t.z = 0;
//    t.w = 0;
    params.image[launch_index.y * params.width + launch_index.x] = t;
}

extern "C" __global__ void __closesthit__scene() {

}

extern "C" __global__ void __anyhit__scene() {

}