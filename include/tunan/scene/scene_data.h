//
// Created by StormPhoenix on 2021/6/1.
//

#ifndef TUNAN_SCENE_DATA_H
#define TUNAN_SCENE_DATA_H

#include <tunan/material/Material.h>
#include <tunan/math.h>
#include <tunan/common.h>
#include <tunan/base/transform.h>

#include <string>
#include <vector>
#include <map>

namespace RENDER_NAMESPACE {
    using namespace material;

    typedef struct ShapeEntity {
        ShapeEntity() = default;

        // Triangles
        std::vector<Point3F> vertices;
        std::vector<Normal3F> normals;
        std::vector<Point2F> texcoords;
        std::vector<int> vertexIndices;
        std::vector<int> normalIndices;
        std::vector<int> texcoordIndices;

        bool faceNormal = false;
        base::Transform toWorld;
        std::string materialName = "";
        int materialIndex = -1;
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
    } SceneData;

    class TriangleMesh {
    public:
        TriangleMesh(const ShapeEntity &entity);

        size_t nVertices;
        size_t nTriangles;
        const Point3F *vertices;
        const int *vertexIndices;
    };
}

#endif //TUNAN_SCENE_DATA_H
