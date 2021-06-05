//
// Created by StormPhoenix on 2021/6/5.
//

#include <tunan/scene/TriangleMesh.h>

namespace RENDER_NAMESPACE {
    TriangleMesh::TriangleMesh(size_t nVertices, Point3F *vertices,
                               size_t nTriangles, int *vertexIndices,
                               Transform &transform) {
        // CUDA memory copy from host to device
        this->nVertices = nVertices;
        this->nTriangles = nTriangles;

        // Vertices
        this->vertices = vertices;

        // Vertex indices
        this->vertexIndices = vertexIndices;

        for (int row = 0; row < 4; row ++) {
            for (int col = 0; col < 4; col ++) {
                transformMatrix[row][col] = transform.mat()[col][row];
            }
        }
    }
}