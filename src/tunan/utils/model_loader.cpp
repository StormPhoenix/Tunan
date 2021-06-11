//
// Created by StormPhoenix on 2021/6/3.
//

#include <tunan/utils/model_loader.h>
#include <tunan/utils/MemoryAllocator.h>
#include <tunan/scene/scene_data.h>

#define TINYOBJLOADER_IMPLEMENTATION

#include <ext/tinyobjloader/tiny_obj_loader.h>

namespace RENDER_NAMESPACE {
    namespace utils {
        bool load_obj(const std::string path, ShapeEntity &entity, MemoryAllocator &allocator) {
            tinyobj::attrib_t attrib;
            std::vector<tinyobj::shape_t> shapes;
            std::vector<tinyobj::material_t> materials;

            std::string warn;
            std::string err;
            bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, path.c_str());
            if (!err.empty()) {
                std::cerr << err << std::endl;
                return false;
            }

            if (!ret) {
                std::cerr << ret << std::endl;
                return false;
            }

            // Vertices
//            vertices.resize(attrib.vertices.size() / 3);
            entity.nVertices = attrib.vertices.size() / 3;
            entity.vertices = allocator.allocateObjects<Point3F>(entity.nVertices);
            for (int i = 0; i < attrib.vertices.size(); i += 3) {
                Float x = attrib.vertices[i];
                Float y = attrib.vertices[i + 1];
                Float z = attrib.vertices[i + 2];
                entity.vertices[i / 3] = {x, y, z};
            }

            // Normals
//            normals.resize(attrib.vertices.size() / 3);
            entity.nNormals = attrib.normals.size() / 3;
            entity.normals = allocator.allocateObjects<Normal3F>(entity.nNormals);
            for (int i = 0; i < attrib.normals.size(); i += 3) {
                Float x = attrib.normals[i];
                Float y = attrib.normals[i + 1];
                Float z = attrib.normals[i + 2];
                entity.normals[i / 3] = {x, y, z};
            }

//            texcoords.resize(attrib.vertices.size() / 2);
            entity.nTexcoords = attrib.texcoords.size() / 2;
            entity.texcoords = allocator.allocateObjects<Point2F>(entity.nTexcoords);
            for (int i = 0; i < attrib.texcoords.size(); i += 2) {
                Float x = attrib.texcoords[i];
                Float y = attrib.texcoords[i + 1];
                entity.texcoords[i / 2] = {x, y};
            }

            ASSERT(shapes.size() == 1, "Only support one shape per *.obj file. ");
            for (int s = 0; s < shapes.size(); s++) {
                int index_offset = 0;
                int materialId = -1;
                entity.nTriangles = shapes[s].mesh.num_face_vertices.size();
                entity.vertexIndices = allocator.allocateObjects<int>(entity.nTriangles * 3);
                entity.normalIndices = allocator.allocateObjects<int>(entity.nTriangles * 3);
                entity.texcoordIndices = allocator.allocateObjects<int>(entity.nTriangles * 3);

                for (int f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
                    int fv = shapes[s].mesh.num_face_vertices[f];
                    ASSERT(fv == 3, "Only support triangle mesh");;

                    for (int v = 0; v < fv; v++) {
                        // access to vertex
                        tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                        entity.vertexIndices[f * 3 + v] = idx.vertex_index;
                        entity.normalIndices[f * 3 + v] = idx.normal_index;
                        entity.texcoordIndices[f * 3 + v] = idx.texcoord_index;
                    }
                    materialId = shapes[s].mesh.material_ids[f];
                    index_offset += fv;
                }
//                entity.materialIndex = materialId;
            }
            return true;
        }

    }
}