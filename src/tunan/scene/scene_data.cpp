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

        for (int row = 0; row < 4; row ++) {
            for (int col = 0; col < 4; col ++) {
                std::cout << entity.toWorld.mat()[col][row] << std::endl;
                transformMatrix[row][col] = entity.toWorld.mat()[col][row];
            }
        }
    }
}