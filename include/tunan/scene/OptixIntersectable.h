//
// Created by StormPhoenix on 2021/5/31.
//

#ifndef TUNAN_OPTIXINTERSECTABLE_H
#define TUNAN_OPTIXINTERSECTABLE_H

#include <tunan/common.h>
#include <tunan/tracer/tracer.h>
#include <tunan/gpu/optix_ray.h>
#include <tunan/base/containers.h>
#include <tunan/material/Material.h>
#include <tunan/scene/scene_data.h>
#include <tunan/scene/SceneIntersectable.h>
#include <tunan/utils/MemoryAllocator.h>

#include <optix.h>
#include <string>
#include <vector>
#include <map>

namespace RENDER_NAMESPACE {
    using material::Material;
    using utils::MemoryAllocator;
    using namespace tracer;

    typedef struct OptiXState {
        RayParams params;
        OptixModule optixModule = nullptr;
        OptixPipeline optixPipeline;
        OptixPipelineCompileOptions pipelineCompileOptions = {0};
        OptixProgramGroup raygenPG = 0;
        // TODO rename
        OptixProgramGroup missPG = 0;
        OptixProgramGroup closesthitPG = 0;
        OptixShaderBindingTable sbt;

        // TODO delete
        CUdeviceptr d_vertices;
        CUdeviceptr d_gas_output_buffer;

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
    typedef SbtRecord<ClosestHitData> ClosestHitRecord;
    typedef SbtRecord<MissData> MissRecord;

    class OptixIntersectable : public SceneIntersectable {
    public:
        OptixIntersectable(SceneData &sceneData, MemoryAllocator &allocator);

        void buildIntersectionStruct(SceneData &sceneData);

        void intersect(RayQueue *rayQueue, MissQueue *missQueue, MaterialEvaQueue *materialEvaQueue,
                       MediaEvaQueue *mediaEvaQueue, AreaLightHitQueue *areaLightQueue,
                       PixelStateArray *pixelStateArray);

        void writeImage() override;

    private:
        void initParams(SceneData &sceneData);

        void createContext();

        void buildAccelStruct(SceneData &sceneData);

        void createModule();

        void createProgramGroups();

        void createSBT();

        void createPipeline();

        OptixTraversableHandle createTriangleGAS(SceneData &data, OptixProgramGroup &closestHitPG);

        OptixTraversableHandle buildBVH(const std::vector<OptixBuildInput> &buildInputs);

    private:
        // TODO delete
        int filmWidth, filmHeight;
        OptiXState state;
        // Memory allocator
        MemoryAllocator &allocator;
        uint64_t buildBVHBytes = 0;
        // Closest hit SBT records
        base::Vector<ClosestHitRecord> closestHitRecords;
    };
}

#endif //TUNAN_OPTIXINTERSECTABLE_H
