//
// Created by StormPhoenix on 2021/6/5.
//

#ifndef TUNAN_MESHES_H
#define TUNAN_MESHES_H

#include <tunan/math.h>
#include <tunan/common.h>
#include <tunan/base/transform.h>
#include <tunan/base/interactions.h>

namespace RENDER_NAMESPACE {
    using namespace base;

    class TriangleMesh {
    public:
        TriangleMesh(size_t nVertices, Point3F *vertices,
                     size_t nNormals, Normal3F *normals,
                     size_t nTexcoords, Point2F *texcoords,
                     size_t nTriangles, int *vertexIndices,
                     int *normalIndices, int *texcoordIndices);

        RENDER_CPU_GPU
        SurfaceInteraction buildSurfaceInteraction(unsigned int index, float b0, float b1, Vector3F wo);

        size_t nVertices;
        Point3F *vertices;

        size_t nNormals;
        Normal3F *normals;

        size_t nTexcoords;
        Point2F *texcoords;

        size_t nTriangles;
        int *vertexIndices;
        int *normalIndices;
        int *texcoordIndices;
    };
}
#endif //TUNAN_MESHES_H
