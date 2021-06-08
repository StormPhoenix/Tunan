//
// Created by StormPhoenix on 2021/6/5.
//

#include <tunan/common.h>
#include <tunan/scene/TriangleMesh.h>

namespace RENDER_NAMESPACE {
    TriangleMesh::TriangleMesh(size_t nVertices, Point3F *vertices, size_t nNormals, Normal3F *normals,
                               size_t nTexcoords, Point2F *texcoords, size_t nTriangles, int *vertexIndices,
                               int *normalIndices, int *texcoordIndices, Transform &transform) :
            nVertices(nVertices), vertices(vertices), nNormals(nNormals), normals(normals),
            nTexcoords(nTexcoords), texcoords(texcoords), nTriangles(nTriangles),
            vertexIndices(vertexIndices), normalIndices(normalIndices), texcoordIndices(texcoordIndices) {
        for (int row = 0; row < 4; row++) {
            for (int col = 0; col < 4; col++) {
                transformMatrix[row][col] = transform.mat()[col][row];
            }
        }
    }

    RENDER_CPU_GPU
    SurfaceInteraction TriangleMesh::buildSurfaceInteraction(unsigned int index, float b0, float b1, Vector3F wo) {
        CHECK(index >= 0 && index < nTriangles);
        int triangleOffset = index * 3;
        Point3F p0 = vertices[vertexIndices[triangleOffset]];
        Point3F p1 = vertices[vertexIndices[triangleOffset + 1]];
        Point3F p2 = vertices[vertexIndices[triangleOffset + 2]];

        Point3F p = p0 * b0 + p1 * b1 + p2 * (1 - b0 - b1);
        Normal3F ng = NORMALIZE(CROSS(p1 - p0, p2 - p0));

        Normal3F ns = ng;
        if (nNormals > 0) {
            Normal3F n0 = normals[normalIndices[triangleOffset]];
            Normal3F n1 = normals[normalIndices[triangleOffset + 1]];
            Normal3F n2 = normals[normalIndices[triangleOffset + 2]];

            ns = n0 * b0 + n1 * b1 + n2 * (1 - b0 - b1);
            if (DOT(ns, ng) < 0.f) {
                ng = -ng;
            }
        }

        Point2F uv(0.);
        if (nTexcoords > 0) {
            Point2F uv0 = texcoords[texcoordIndices[triangleOffset]];
            Point2F uv1 = texcoords[texcoordIndices[triangleOffset + 1]];
            Point2F uv2 = texcoords[texcoordIndices[triangleOffset + 2]];

            uv = uv0 * b0 + uv1 * b1 + uv2 * (1 - b0 - b1);
        }

        // error offset computation
        Float xError = (std::abs(b0 * p0.x) + std::abs(b1 * p1.x) + std::abs((1 - b0 - b1) * p2.x))
                       * math::gamma(7);
        Float yError = (std::abs(b0 * p0.y) + std::abs(b1 * p1.y) + std::abs((1 - b0 - b1) * p2.y))
                       * math::gamma(7);
        Float zError = (std::abs(b0 * p0.z) + std::abs(b1 * p1.z) + std::abs((1 - b0 - b1) * p2.z))
                       * math::gamma(7);
        Vector3F error = Vector3F(xError, yError, zError);

        return SurfaceInteraction(p, ng, ns, wo, uv, error);
    }
}