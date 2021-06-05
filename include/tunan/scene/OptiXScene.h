//
// Created by StormPhoenix on 2021/5/31.
//

#ifndef TUNAN_OPTIXSCENE_H
#define TUNAN_OPTIXSCENE_H

#include <tunan/common.h>
#include <tunan/base/containers.h>
#include <tunan/scene/scene_data.h>
#include <tunan/material/Material.h>
#include <tunan/utils/MemoryAllocator.h>
#include <tunan/gpu/optix_ray.h>

#include <optix.h>

#include <string>
#include <vector>
#include <map>

namespace RENDER_NAMESPACE {
    using material::Material;
    using utils::MemoryAllocator;

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

typedef SbtRecord <RayGenData> RaygenRecord;
typedef SbtRecord <ClosestHitData> ClosestHitRecord;
typedef SbtRecord <MissData> MissRecord;

class OptiXScene {
public:
    OptiXScene(SceneData &sceneData, MemoryAllocator &allocator);

    void buildIntersectionStruct(const SceneData &sceneData);

    void intersect();

private:
    void initParams(const SceneData &sceneData);

    void createContext();

    void buildAccelStruct(const SceneData &sceneData);

    void createModule();

    void createProgramGroups();

    void createSBT();

    void createPipeline();

    OptixTraversableHandle createTriangleGAS(const SceneData &data, OptixProgramGroup &closestHitPG);

    OptixTraversableHandle buildBVH(const std::vector<OptixBuildInput> &buildInputs);

private:
    int filmWidth, filmHeight;
    OptiXState state;
    // Memory allocator
    MemoryAllocator &allocator;
    uint64_t buildBVHBytes = 0;
    // Closest hit SBT records
    base::Vector <ClosestHitRecord> closestHitRecords;
};

}

#endif //TUNAN_OPTIXSCENE_H
