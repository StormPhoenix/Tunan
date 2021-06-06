#include <optix.h>
#include <tunan/common.h>
#include <tunan/gpu/optix_ray.h>

// *.cu file cannot link these variables functions and classes
#include <tunan/scene/Camera.cpp>
#include <tunan/scene/Ray.cpp>
#include <tunan/base/transform.cpp>

using namespace RENDER_NAMESPACE;

extern "C" {
__constant__ RayParams params;
}

extern "C" __global__ void __raygen__findclosesthit() {

    uint3 launch_index = optixGetLaunchIndex();
    unsigned int pixel_x = launch_index.x;
    unsigned int pixel_y = launch_index.y;
//    Camera *camera = params.camera;
//    int image_width = camera->getWidth();
//    int image_height = camera->getHeight();

    // TODO delete
    uchar3 t;
    t.x = 0;
    t.y = 0;
    t.z = 0;
//    t.w = 0;
//    params.outputImage[launch_index.y * image_width + launch_index.x] = t;

//    Ray ray =  camera->generateRay(pixel_x, pixel_y);
//    float3 ray_origin = make_float3(ray.getOrigin().x, ray.getOrigin().y, ray.getOrigin().z);
//    float3 ray_direction = make_float3(ray.getDirection().x, ray.getDirection().y, ray.getDirection().z);

//    float tMin = 0.f;
//    float tMax = 1e30f;
//    optixTrace(params.traversable, ray_origin, ray_direction, tMin, tMax, 0.0f,
//               OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE, 0, 1, 0);
}

extern "C" __global__ void __closesthit__scene() {
//    uint3 launch_index = optixGetLaunchIndex();
//    unsigned int pixel_x = launch_index.x;
//    unsigned int pixel_y = launch_index.y;

//    Camera *camera = params.camera;
//    int image_width = camera->getWidth();
//    int image_height = camera->getHeight();

//    unsigned int tri_index = optixGetPrimitiveIndex();

    // TODO delete
//    uchar3 t;
//    t.x = 0;
//    t.y = 0;
//    t.z = tri_index * 255 / 8;
//    t.w = 0;
//    params.outputImage[launch_index.y * image_width + launch_index.x] = t;
}

extern "C" __global__ void __miss__findclosehit_scene() {
    uint3 launch_index = optixGetLaunchIndex();
    unsigned int pixel_x = launch_index.x;
    unsigned int pixel_y = launch_index.y;

//    Camera *camera = params.camera;
//    int image_width = camera->getWidth();
//    int image_height = camera->getHeight();

    // TODO delete
    uchar3 t;
    t.x = 100;
    t.y = 0;
    t.z = 0;
//    t.w = 0;
//    params.outputImage[launch_index.y * image_width + launch_index.x] = t;
}

extern "C" __global__ void __anyhit__scene() {
    uint3 launch_index = optixGetLaunchIndex();
    unsigned int pixel_x = launch_index.x;
    unsigned int pixel_y = launch_index.y;

//    Camera *camera = params.camera;
//    int image_width = camera->getWidth();
//    int image_height = camera->getHeight();

    // TODO delete
    uchar3 t;
    t.x = 200;
    t.y = 0;
    t.z = 0;
//    t.w = 0;
//    params.outputImage[launch_index.y * image_width + launch_index.x] = t;
}