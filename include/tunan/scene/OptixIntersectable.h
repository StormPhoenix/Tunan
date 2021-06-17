//
// Created by StormPhoenix on 2021/5/31.
//

#ifndef TUNAN_OPTIXINTERSECTABLE_H
#define TUNAN_OPTIXINTERSECTABLE_H

#include <tunan/common.h>
#include <tunan/tracer/tracer.h>
#include <tunan/gpu/optix_ray.h>
#include <tunan/base/containers.h>
#include <tunan/material/materials.h>
#include <tunan/scene/scenedata.h>
#include <tunan/scene/SceneIntersectable.h>
#include <tunan/utils/ResourceManager.h>

#include <optix.h>
#include <string>
#include <vector>
#include <map>

namespace RENDER_NAMESPACE {
    using material::Material;
    using utils::ResourceManager;
    using namespace tracer;

    typedef struct OptiXState {
        RayParams params;
        OptixModule optixModule = nullptr;
        OptixPipeline optixPipeline;
        OptixPipelineCompileOptions pipelineCompileOptions = {0};
        OptixProgramGroup raygenPG = 0;
        OptixProgramGroup missPG = 0;
        OptixProgramGroup closestHitPG = 0;

        OptixProgramGroup raygenShadowRayPG = 0;
        OptixProgramGroup anyHitShadowRayPG = 0;
        OptixProgramGroup missShadowRayPG = 0;

        OptixShaderBindingTable closeHitSbt;
        OptixShaderBindingTable shadowRaySbt;

        OptixDeviceContext optixContext;
        CUstream cudaStream = 0;
    } OptiXState;

    template<typename T>
    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecord {
        __align__(OPTIX_SBT_RECORD_ALIGNMENT)
        char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        T data;
    };

    typedef SbtRecord<RayGenData> RaygenRecord;
    typedef SbtRecord<ClosestHitData> HitRecord;
    typedef SbtRecord<MissData> MissRecord;

    class OptixIntersectable : public SceneIntersectable {
    public:
        OptixIntersectable(SceneData &sceneData, ResourceManager *allocator);

        void buildIntersectionStruct(SceneData &sceneData);

        void findClosestHit(RayQueue *rayQueue, MissQueue *missQueue, MaterialEvaQueue *materialEvaQueue,
                            MediaEvaQueue *mediaEvaQueue, AreaLightHitQueue *areaLightQueue,
                            PixelStateArray *pixelStateArray) override;

        void traceShadowRay(ShadowRayQueue *shadowRayQueue, PixelStateArray *pixelStateArray) override;

    private:
        void initParams(SceneData &sceneData);

        void createContext();

        void buildAccelStruct(SceneData &sceneData);

        void createModule();

        void createProgramGroups();

        void createSBT();

        void createPipeline();

        OptixTraversableHandle createTriangleGAS(SceneData &data, OptixProgramGroup &closestHitPG,
                                                 OptixProgramGroup &shadowRayHitPG);

        OptixTraversableHandle buildBVH(const std::vector<OptixBuildInput> &buildInputs);

    private:
        OptiXState state;
        ResourceManager *allocator;
        uint64_t buildBVHBytes = 0;
        // SBT records
        base::Vector<HitRecord> closestHitRecords;
        base::Vector<HitRecord> shadowRayRecords;
    };
}

#endif //TUNAN_OPTIXINTERSECTABLE_H
