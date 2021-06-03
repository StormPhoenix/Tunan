//
// Created by StormPhoenix on 2021/6/1.
//

#include <tunan/scene/scene_data.h>
#include <iostream>

namespace RENDER_NAMESPACE {
    TriangleMesh::TriangleMesh(const ShapeEntity &entity) {
        ASSERT(entity.vertexIndices.size() % 3 == 0, "TriangleMesh error: tirangles should align to 3. ");
        // CUDA memory copy from host to device
        nVertices = entity.vertices.size();
        nTriangles = entity.vertexIndices.size() / 3;

        // Vertices
        vertices = entity.vertices.data();

        // Vertex indices
        vertexIndices = entity.vertexIndices.data();
    }
}