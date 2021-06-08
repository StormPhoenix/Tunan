//
// Created by StormPhoenix on 2021/6/1.
//

#include <tunan/scene/scene_data.h>
#include <tunan/utils/MemoryAllocator.h>

namespace RENDER_NAMESPACE {

    void ShapeEntity::createAreaLights(const Spectrum &radiance, utils::MemoryAllocator &allocator) {
        for (int i = 0; i < nTriangles; i++) {
            int offset = i * 3;
            Point3F p0 = vertices[vertexIndices[offset]];
            Point3F p1 = vertices[vertexIndices[offset + 1]];
            Point3F p2 = vertices[vertexIndices[offset + 2]];

            Normal3F n0(0), n1(0), n2(0);
            if (nNormals > 0) {
                n0 = normals[normalIndices[offset]];
                n1 = normals[normalIndices[offset + 1]];
                n2 = normals[normalIndices[offset + 2]];
            }
            Shape shape = allocator.newObject<Triangle>(p0, p1, p2, n0, n1, n2);
            // TODO medium
            DiffuseAreaLight *areaLight = allocator.newObject<DiffuseAreaLight>(radiance, shape, nullptr);
            areaLights.push_back(areaLight);
        }
    }
}