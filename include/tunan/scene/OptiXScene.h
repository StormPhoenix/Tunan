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
        OptixProgramGroup raygenFindIntersectionProgramGroup = 0;
        // TODO rename
        OptixProgramGroup missProgramGroup = 0;
        OptixProgramGroup closestHitProgramGroup = 0;
        OptixShaderBindingTable sbt;

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
// TODO delete for testing
typedef SbtRecord <ClosestHitData> ClosestHitRecord;
// TODO
//    struct ClosestHitRecord;

class OptiXScene {
public:
    OptiXScene(SceneData &sceneData, MemoryAllocator &allocator);

    void buildOptiXData(const SceneData &sceneData);

    void intersect();

private:
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
