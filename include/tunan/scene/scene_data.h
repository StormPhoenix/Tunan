//
// Created by StormPhoenix on 2021/6/1.
//

#ifndef TUNAN_SCENE_DATA_H
#define TUNAN_SCENE_DATA_H

#include <tunan/material/Material.h>
#include <tunan/math.h>
#include <tunan/common.h>

#include <string>
#include <vector>
#include <map>

namespace RENDER_NAMESPACE {
    using namespace material;

    typedef struct ShapeEntity {
        // Triangles
        std::vector<Point3F> vertices;
        std::vector<Normal3F> normals;
        std::vector<Point2F> uvs;
        std::vector<int> indices;

        std::string materialName;
        int materialIndex;
    } ShapeEntity;

    typedef struct SceneData {
        // Entities
        std::vector<ShapeEntity> entities;

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
        const Normal3F *normals;
        const Point2F *uvs;
        const int *indices;
    };
}

#endif //TUNAN_SCENE_DATA_H
