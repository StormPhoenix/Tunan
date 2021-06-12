//
// Created by StormPhoenix on 2021/6/1.
//

#ifndef TUNAN_SCENEDATA_H
#define TUNAN_SCENEDATA_H

#include <tunan/math.h>
#include <tunan/common.h>
#include <tunan/scene/lights.h>
#include <tunan/base/transform.h>
#include <tunan/base/containers.h>
#include <tunan/material/materials.h>
#include <tunan/utils/MemoryAllocator.h>

#include <string>
#include <vector>
#include <map>

namespace RENDER_NAMESPACE {
    using namespace material;

    typedef struct ShapeEntity {
        ShapeEntity() = default;

        void createAreaLights(const Spectrum &radiance, utils::MemoryAllocator &allocator);

        int nVertices = 0;
        Point3F *vertices = nullptr;

        int nNormals = 0;
        Normal3F *normals = nullptr;

        int nTexcoords = 0;
        Point2F *texcoords = nullptr;

        int nTriangles = 0;
        int *vertexIndices = nullptr;
        int *normalIndices = nullptr;
        int *texcoordIndices = nullptr;

        bool faceNormal = false;
        Material material;
        base::Transform toWorld;
        DiffuseAreaLight *areaLights = nullptr;
//        std::string materialName = "";
//        int materialIndex = -1;
    } ShapeEntity;

    typedef struct SceneData {
        std::string sceneDirectory;

        // Entities
        std::vector<ShapeEntity> entities;

        // Film params
        // TODO check variable meaning
        std::string filmType;
        std::string filename = "render";
        int width = 300;
        int height = 300;
        std::string fileFormat = "png";
        std::string pixelFormat = "";
        float gamma = 2;
        bool banner = false;
        std::string rfilter = "";

        // Camera params
        // TODO check variable meaning
        std::string type = "perspective";
        float fov = 45.0f;
        base::Transform cameraToWorld;

        // Integrator params
        // TODO check variable meaning
        std::string integratorType = "path";
        int maxDepth = 12;
        int sampleNum = 100;
        int delta = 1;
        bool strictNormals = false;
        int photonCount = 1024;
        float initialRadius = 0.03;
        float radiusDecay = 0.3;

        // Materials
        std::map<std::string, Material> namedMaterial;

        std::vector<Material> materials;
        std::map<std::string, Material> materialMap;
        base::Vector<Light> *lights = nullptr;
    } SceneData;
}

#endif //TUNAN_SCENEDATA_H
