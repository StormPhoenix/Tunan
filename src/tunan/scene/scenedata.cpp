//
// Created by StormPhoenix on 2021/6/1.
//

#include <tunan/scene/scenedata.h>
#include <tunan/utils/ResourceManager.h>

namespace RENDER_NAMESPACE {

    void ShapeEntity::createAreaLights(const Spectrum &radiance, utils::ResourceManager *allocator) {
        if (nTriangles > 0) {
            areaLights = allocator->allocateObjects<DiffuseAreaLight>(nTriangles);
        }

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
            Shape shape = allocator->newObject<Triangle>(p0, p1, p2, n0, n1, n2);
            // TODO medium
            areaLights[i] = DiffuseAreaLight(radiance, shape, MediumInterface());
        }
    }

    void ShapeEntity::computeWorldBound() {
        Float minX, minY, minZ;
        Float maxX, maxY, maxZ;

        if (nVertices > 0) {
            maxX = minX = vertices[0].x;
            maxY = minY = vertices[0].y;
            maxZ = minZ = vertices[0].z;

            for (int i = 1; i < nVertices; i++) {
                maxX = maxX >= vertices[i].x ? maxX : vertices[i].x;
                minX = minX <= vertices[i].x ? minX : vertices[i].x;

                maxY = maxY >= vertices[i].y ? maxY : vertices[i].y;
                minY = minY <= vertices[i].y ? minY : vertices[i].y;

                maxZ = maxZ >= vertices[i].z ? maxZ : vertices[i].z;
                minZ = minZ <= vertices[i].z ? minZ : vertices[i].z;
            }
        }

        minVertex = Point3F(minX, minY, minZ);
        maxVertex = Point3F(maxX, maxY, maxZ);
    }

    void SceneData::computeWorldBound() {
        if (entities.size() > 0) {
            for (int i = 0; i < entities.size(); i++) {
                entities[i].computeWorldBound();
            }

            worldMin = entities[0].minVertex;
            worldMax = entities[0].maxVertex;

            for (int i = 1; i < entities.size(); i++) {
                worldMin.x = worldMin.x < entities[i].minVertex.x ? worldMin.x : entities[i].minVertex.x;
                worldMin.y = worldMin.y < entities[i].minVertex.y ? worldMin.y : entities[i].minVertex.y;
                worldMin.z = worldMin.z < entities[i].minVertex.z ? worldMin.z : entities[i].minVertex.z;

                worldMax.x = worldMax.x > entities[i].maxVertex.x ? worldMax.x : entities[i].maxVertex.x;
                worldMax.y = worldMax.y > entities[i].maxVertex.y ? worldMax.y : entities[i].maxVertex.y;
                worldMax.z = worldMax.z > entities[i].maxVertex.z ? worldMax.z : entities[i].maxVertex.z;
            }

            worldMin = Point3F(worldMin.x - 0.0001, worldMin.y - 0.0001, worldMin.z - 0.0001);
            worldMax = Point3F(worldMax.x + 0.0001, worldMax.y + 0.0001, worldMax.z + 0.0001);
        }
    }

    void SceneData::postprocess() {
        computeWorldBound();
        if (envLights != nullptr) {
            for (int i = 0; i < envLights->size(); i ++) {
                (*envLights)[i]->worldBound(worldMin, worldMax);
            }
        }
    }
}