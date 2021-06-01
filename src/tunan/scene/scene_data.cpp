//
// Created by StormPhoenix on 2021/6/1.
//

#include <tunan/scene/scene_data.h>
#include <iostream>

namespace RENDER_NAMESPACE {
    TriangleMesh::TriangleMesh(const ShapeEntity &entity) {
        ASSERT(entity.indices.size() % 3 == 0, "TriangleMesh error: tirangles should align to 3. ");
        // CUDA memory copy from host to device
        nVertices = entity.vertices.size();
        nTriangles = entity.indices.size() / 3;
        vertices = entity.vertices.data();
        normals = entity.normals.data();
        uvs = entity.uvs.data();
        indices = entity.indices.data();
    }
}