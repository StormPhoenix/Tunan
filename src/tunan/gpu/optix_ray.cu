#include <optix.h>
#include <tunan/common.h>
#include <tunan/gpu/optix_ray.h>

// *.cu file cannot link these variables functions and classes
#include <tunan/scene/cameras.cpp>
#include <tunan/scene/Ray.cpp>
#include <tunan/scene/meshes.cpp>
#include <tunan/base/transform.cpp>
#include <tunan/base/interactions.cpp>

using namespace RENDER_NAMESPACE;

extern "C" {
__constant__ RayParams params;
}

extern "C" __global__ void __raygen__findClosestHit() {

    uint3 launchIndex = optixGetLaunchIndex();
    RayDetails &r = (*params.rayQueue)[launchIndex.x];
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
    SurfaceInteraction si = data->mesh->buildSurfaceInteraction(triangleIndex, 1 - barycentric.x - barycentric.y,
                                                                barycentric.x, -r.ray.getDirection());

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
}

extern "C" __global__ void __miss__findclosesthit() {
    optixSetPayload_0(1);
}

extern "C" __global__ void __anyhit__scene() {
    uint3 launch_index = optixGetLaunchIndex();
}

extern "C" __device__ float misWeight(int nSampleF, Float pdfF, int nSampleG, Float pdfG) {
    Float f = nSampleF * pdfF;
    Float g = nSampleG * pdfG;
    return (f * f) / (g * g + f * f);
}

extern "C" __global__ void __raygen__shadowRay() {
    uint3 launchIndex = optixGetLaunchIndex();
    ShadowRayDetails &r = (*params.shadowRayQueue)[launchIndex.x];

    // Tracing
    unsigned int missed = 0;
    float tMin = 1e-5f;
    float tMax = r.tMax;
    float3 ray_origin = make_float3(r.ray.getOrigin().x, r.ray.getOrigin().y, r.ray.getOrigin().z);
    float3 ray_direction = make_float3(r.ray.getDirection().x, r.ray.getDirection().y, r.ray.getDirection().z);

    optixTrace(params.traversable, ray_origin, ray_direction, tMin, tMax, 0.0f,
               OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE, 0, 1, 0,
               missed);

    if (missed) {
        // TODO handle medium transmittance
        Spectrum L(0);
        if (r.deltaType) {
            L = r.beta * r.L / r.sampleLightPdf;
        } else {
            float weight = misWeight(1, r.sampleLightPdf, 1, r.scatterPdf);
            L = r.beta * r.L * weight / r.sampleLightPdf;
        }

        PixelState &state = (*params.pixelStateArray)[r.pixelIndex];
        state.L += L / r.lightPdf;
    }
}

extern "C" __global__ void __anyhit__shadowRay() {
    // Do nothing
}

extern "C" __global__ void __miss__shadowRay() {
    optixSetPayload_0(1);
}