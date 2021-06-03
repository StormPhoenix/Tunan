//
// Created by StormPhoenix on 2021/6/1.
//

#include <tunan/scene/scene_data.h>
#include <iostream>

namespace RENDER_NAMESPACE {
    TriangleMesh::TriangleMesh(const ShapeEntity &entity) {
        // CUDA memory copy from host to device
        nVertices = entity.nVertices;
        nTriangles = entity.nTriangles;

        // Vertices
        vertices = entity.vertices;

        // Vertex indices
        vertexIndices = entity.vertexIndices;
    }
}