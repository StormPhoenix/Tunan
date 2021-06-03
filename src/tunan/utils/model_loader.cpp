//
// Created by StormPhoenix on 2021/6/3.
//

#include <tunan/utils/model_loader.h>
#include <tunan/scene/scene_data.h>

#define TINYOBJLOADER_IMPLEMENTATION

#include <ext/tinyobjloader/tiny_obj_loader.h>

namespace RENDER_NAMESPACE {
    namespace utils {
        bool load_obj(const std::string path, ShapeEntity &entity) {
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
            for (int i = 0; i < attrib.vertices.size(); i += 3) {
                Float x = attrib.vertices[i];
                Float y = attrib.vertices[i + 1];
                Float z = attrib.vertices[i + 2];
                entity.vertices.push_back({x, y, z});
            }

            // Normals
//            normals.resize(attrib.vertices.size() / 3);
            for (int i = 0; i < attrib.normals.size(); i += 3) {
                Float x = attrib.normals[i];
                Float y = attrib.normals[i + 1];
                Float z = attrib.normals[i + 2];
                entity.normals.push_back({x, y, z});
            }

//            texcoords.resize(attrib.vertices.size() / 2);
            for (int i = 0; i < attrib.texcoords.size(); i += 2) {
                Float x = attrib.texcoords[i];
                Float y = attrib.texcoords[i + 1];
                entity.texcoords.push_back({x, y});
            }

            ASSERT(shapes.size() == 1, "Only support one shape per *.obj file. ");
            for (int s = 0; s < shapes.size(); s++) {
                int index_offset = 0;
                int materialId = -1;
                for (int f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
                    int fv = shapes[s].mesh.num_face_vertices[f];
                    ASSERT(fv == 3, "Only support triangle mesh");;

                    for (int v = 0; v < fv; v++) {
                        // access to vertex
                        tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                        entity.vertexIndices.push_back(idx.vertex_index);
                        entity.normalIndices.push_back(idx.normal_index);
                        entity.texcoordIndices.push_back(idx.texcoord_index);
                    }
                    materialId = shapes[s].mesh.material_ids[f];
                    index_offset += fv;
                }
                entity.materialIndex = materialId;
            }
            return true;
        }

    }
}