#include <optix.h>
#include <tunan/common.h>
#include <tunan/gpu/optix_ray.h>

// *.cu file cannot link these variables functions and classes
#include <tunan/scene/Camera.cpp>
#include <tunan/scene/Ray.cpp>
#include <tunan/scene/TriangleMesh.cpp>
#include <tunan/base/transform.cpp>
#include <tunan/base/interactions.cpp>

using namespace RENDER_NAMESPACE;

extern "C" {
__constant__ RayParams params;
}

extern "C" __global__ void __raygen__findclosesthit() {

    uint3 launch_index = optixGetLaunchIndex();
    RayDetails &r = (*params.rayQueue)[launch_index.x];
    float3 ray_origin = make_float3(r.ray.getOrigin().x, r.ray.getOrigin().y, r.ray.getOrigin().z);
    float3 ray_direction = make_float3(r.ray.getDirection().x, r.ray.getDirection().y, r.ray.getDirection().z);

    float tMin = 0.f;
    float tMax = 1e30f;

    unsigned int missed = 0;
    optixTrace(params.traversable, ray_origin, ray_direction, tMin, tMax, 0.0f,
               OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE, 0, 1, 0,
               missed);

    if (missed) {
        MissQueue *mq = params.missQueue;
        mq->enqueue(r);
    }
}

extern "C" __global__ void __closesthit__scene() {
    unsigned int triangleIndex = optixGetPrimitiveIndex();
    float2 barycentric = optixGetTriangleBarycentrics();
    RayDetails &r = (*params.rayQueue)[optixGetLaunchIndex().x];

    const ClosestHitData *data = (const ClosestHitData *) optixGetSbtDataPointer();
    SurfaceInteraction si = data->mesh->buildSurfaceInteraction(triangleIndex, barycentric.x,
                                                                barycentric.y, -r.ray.getDirection());

    if (data->areaLights != nullptr) {
        AreaLightHitDetails areaLightHitDetails;
        areaLightHitDetails.areaLight = data->areaLights + triangleIndex;
        areaLightHitDetails.pixelIndex = r.pixelIndex;
        areaLightHitDetails.bounce = r.bounce;
        areaLightHitDetails.si = si;
        areaLightHitDetails.specularBounce = r.specularBounce;
        params.areaLightQueue->enqueue(areaLightHitDetails);
    }

    MaterialEvaDetails materialEvaDetails;
    materialEvaDetails.bounce = r.bounce;
    materialEvaDetails.pixelIndex = r.pixelIndex;
    materialEvaDetails.si = si;
    materialEvaDetails.material = data->material;

    params.materialEvaQueue->enqueue(materialEvaDetails);

    // TODO for testing
    uchar3 t;
    t.x = int((si.ng.x + 1.0) * 127.5);
    t.y = int((si.ng.y + 1.0) * 127.5);
    t.z = int((si.ng.z + 1.0) * 127.5);
//    t.x = 255;
//    t.y = 0;
//    t.z = 0;
    PixelState &state = (*params.pixelStateArray)[r.pixelIndex];
//    params.outputImage[state.pixelY * 800 + state.pixelX] = t;
    params.outputImage[state.pixelY * 1024 + state.pixelX] = t;
}

extern "C" __global__ void __miss__findclosesthit() {
    optixSetPayload_0(1);

    // TODO for testing
//    RayDetails &r = (*params.rayQueue)[optixGetLaunchIndex().x];
//    uchar3 t;
//    t.x = 0;
//    t.y = 255;
//    t.z = 0;
//    PixelState &state = (*params.pixelStateArray)[r.pixelIndex];
//    params.outputImage[state.pixelY * 800 + state.pixelX] = t;
//    params.outputImage[state.pixelY * 1024 + state.pixelX] = t;
}

extern "C" __global__ void __anyhit__scene() {
    uint3 launch_index = optixGetLaunchIndex();
//    Camera *camera = params.camera;
//    int image_width = camera->getWidth();
//    int image_height = camera->getHeight();

    // TODO delete
//    uchar3 t;
//    t.x = 200;
//    t.y = 0;
//    t.z = 0;
//    t.w = 0;
//    params.outputImage[launch_index.y * image_width + launch_index.x] = t;
}