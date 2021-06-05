//
// Created by StormPhoenix on 2021/6/5.
//

#ifndef TUNAN_TRIANGLEMESH_H
#define TUNAN_TRIANGLEMESH_H

#include <tunan/math.h>
#include <tunan/common.h>
#include <tunan/base/transform.h>

namespace RENDER_NAMESPACE {
    using base::Transform;

    class TriangleMesh {
    public:
        TriangleMesh(size_t nVertices, Point3F *vertices,
                     size_t nTriangles, int *vertexIndices,
                     Transform &transform);

        size_t nVertices;
        size_t nTriangles;
        const Point3F *vertices;
        const int *vertexIndices;

        Float transformMatrix[4][4];
    };
}
#endif //TUNAN_TRIANGLEMESH_H
