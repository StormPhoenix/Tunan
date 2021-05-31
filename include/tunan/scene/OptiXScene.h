//
// Created by Graphics on 2021/5/31.
//

#ifndef TUNAN_OPTIXSCENE_H
#define TUNAN_OPTIXSCENE_H

#include <tunan/common.h>
#include <tunan/material/Material.h>
#include <tunan/utils/MemoryAllocator.h>

#include <optix.h>

#include <string>
#include <vector>
#include <map>

namespace RENDER_NAMESPACE {
    using material::Material;
    using utils::MemoryAllocator;

    typedef struct OptiXState {
        OptixDeviceContext optixContext;
        OptixModule optixModule = nullptr;
    } OptiXState;

    class OptiXScene {
    public:
        void buildOptiXData();

    private:
        OptiXState state;
        MemoryAllocator &allocator;
        // Materials
        std::map<std::string, Material> namedMaterial;
        std::vector<Material> materials;
    };
}

#endif //TUNAN_OPTIXSCENE_H
