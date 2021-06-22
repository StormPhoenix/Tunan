//
// Created by StormPhoenix on 2021/6/2.
//

#include <tunan/scene/importers.h>
#include <tunan/base/transform.h>
#include <tunan/base/spectrum.h>
#include <tunan/utils/model_loader.h>
#include <tunan/utils/ResourceManager.h>
#include <tunan/scene/scenedata.h>
#include <tunan/material/microfacets.h>
#include <tunan/material/materials.h>

#include <ext/pugixml/pugixml.hpp>

#include <fstream>
#include <sstream>

namespace RENDER_NAMESPACE {
    namespace importer {
        using utils::ResourceManager;
        using namespace material;
        using namespace microfacet;

#define GET_PARSE_INFO_VALUE_FUNC_DECLARE(Type, TypeUpperCase) \
    Type get##TypeUpperCase##Value(const std::string name, const Type defaultValue);                                                               \

#define SET_PARSE_INFO_VALUE_FUNC_DECLARE(Type, TypeUpperCase) \
    void set##TypeUpperCase##Value(const std::string name, const Type value);

#define GET_PARSE_INFO_VALUE_FUNC_DEFINE(Type, TypeUpperCase, TypeLowerCase) \
    Type XmlParseInfo::get##TypeUpperCase##Value(const std::string name, const Type defaultValue) { \
        if (container.count(name) > 0) {          \
            return container[name].value.TypeLowerCase##Value;                                        \
        } else {                                  \
            return defaultValue;                                          \
        }                                         \
    }

#define SET_PARSE_INFO_VALUE_FUNC_DEFINE(Type, TypeUpperCase, TypeLowerCase) \
    void XmlParseInfo::set##TypeUpperCase##Value(const std::string name, const Type value) {\
        XmlAttrVal &attrVal = container[name];               \
        attrVal.type = XmlAttrVal::Attr_##TypeUpperCase;                     \
        attrVal.value.TypeLowerCase##Value = value;                          \
    }
        using namespace base;

        typedef struct XmlAttrVal {
            enum AttrType {
                Attr_Spectrum,
                Attr_Bool,
                Attr_Int,
                Attr_Float,
                Attr_Transform,
                Attr_String,
                Attr_Vector,
                Attr_SpectrumTexture,
                Attr_None,
            } type;

            struct Val {
                Spectrum spectrumValue;
                SpectrumTexture spectrumTextureValue;
                bool boolValue;
                int intValue;
                Float floatValue;
                Transform transformValue;
                std::string stringValue;
                Vector3F vectorValue;
            } value;

            XmlAttrVal() {}
        } XmlAttrVal;

        typedef struct XmlParseInfo {
            Transform transformMat;
            Material currentMaterial;
            bool hasAreaLight = false;

            Medium currentExteriorMedium;
            Medium currentInteriorMedium;

            XmlAttrVal get(std::string name) {
                return container[name];
            }

            XmlAttrVal::AttrType getType(std::string name) {
                if (container.count(name) > 0) {
                    return container[name].type;
                } else {
                    return XmlAttrVal::Attr_None;
                }
            }

            void set(std::string name, XmlAttrVal value) {
                container[name] = value;
            }

            bool attrExists(std::string name) {
                return container.count(name) > 0;
            }

            GET_PARSE_INFO_VALUE_FUNC_DECLARE(bool, Bool)

            SET_PARSE_INFO_VALUE_FUNC_DECLARE(bool, Bool)

            GET_PARSE_INFO_VALUE_FUNC_DECLARE(int, Int)

            SET_PARSE_INFO_VALUE_FUNC_DECLARE(int, Int)

            GET_PARSE_INFO_VALUE_FUNC_DECLARE(Spectrum, Spectrum)

            SET_PARSE_INFO_VALUE_FUNC_DECLARE(Spectrum, Spectrum)

            GET_PARSE_INFO_VALUE_FUNC_DECLARE(SpectrumTexture, SpectrumTexture)

            SET_PARSE_INFO_VALUE_FUNC_DECLARE(SpectrumTexture, SpectrumTexture)

            GET_PARSE_INFO_VALUE_FUNC_DECLARE(float, Float)

            SET_PARSE_INFO_VALUE_FUNC_DECLARE(float, Float)

            GET_PARSE_INFO_VALUE_FUNC_DECLARE(Transform, Transform)

            SET_PARSE_INFO_VALUE_FUNC_DECLARE(Transform, Transform)

            GET_PARSE_INFO_VALUE_FUNC_DECLARE(std::string, String)

            SET_PARSE_INFO_VALUE_FUNC_DECLARE(std::string, String)

            SET_PARSE_INFO_VALUE_FUNC_DECLARE(Vector3F, Vector)

            GET_PARSE_INFO_VALUE_FUNC_DECLARE(Vector3F, Vector)

        private:
            std::map <std::string, XmlAttrVal> container;
        } XmlParseInfo;

        GET_PARSE_INFO_VALUE_FUNC_DEFINE(bool, Bool, bool);

        SET_PARSE_INFO_VALUE_FUNC_DEFINE(bool, Bool, bool);

        GET_PARSE_INFO_VALUE_FUNC_DEFINE(int, Int, int);

        SET_PARSE_INFO_VALUE_FUNC_DEFINE(int, Int, int);

        GET_PARSE_INFO_VALUE_FUNC_DEFINE(Spectrum, Spectrum, spectrum);

        SET_PARSE_INFO_VALUE_FUNC_DEFINE(Spectrum, Spectrum, spectrum);

        GET_PARSE_INFO_VALUE_FUNC_DEFINE(SpectrumTexture, SpectrumTexture, spectrumTexture);

        SET_PARSE_INFO_VALUE_FUNC_DEFINE(SpectrumTexture, SpectrumTexture, spectrumTexture);

        GET_PARSE_INFO_VALUE_FUNC_DEFINE(float, Float, float);

        SET_PARSE_INFO_VALUE_FUNC_DEFINE(float, Float, float);

        GET_PARSE_INFO_VALUE_FUNC_DEFINE(Transform, Transform, transform);

        SET_PARSE_INFO_VALUE_FUNC_DEFINE(Transform, Transform, transform);

        GET_PARSE_INFO_VALUE_FUNC_DEFINE(std::string, String, string);

        SET_PARSE_INFO_VALUE_FUNC_DEFINE(std::string, String, string);

        SET_PARSE_INFO_VALUE_FUNC_DEFINE(Vector3F, Vector, vector);

        GET_PARSE_INFO_VALUE_FUNC_DEFINE(Vector3F, Vector, vector);

        typedef enum TagType {
            Tag_Scene,
            Tag_Mode,

            // Scene components
            Tag_Integrator,
            Tag_Emitter,
            Tag_Sensor,
            Tag_Sampler,
            Tag_Film,
            Tag_Rfilter,
            Tag_Shape,
            Tag_BSDF,
            Tag_Ref,

            Tag_Texture,
            Tag_Medium,
            Tag_Integer,
            Tag_Float,
            Tag_Boolean,
            Tag_String,
            Tag_Transform,
            Tag_Matrix,
            Tag_Vector,
            Tag_LookAt,
            Tag_RGB,
        } TagType;

        static std::map <std::string, TagType> nodeTypeMap;

        inline Vector3F toVector(const std::string &val) {
            Vector3F ret;
            char *tmp;
#if defined(_RENDER_DATA_DOUBLE_)
            ret[0] = strtod(val.c_str(), &tmp);
                for (int i = 1; i < 3; i++) {
                    tmp++;
                    ret[i] = strtod(tmp, &tmp);
                }
#else
            ret[0] = strtof(val.c_str(), &tmp);
            for (int i = 1; i < 3; i++) {
                tmp++;
                ret[i] = strtof(tmp, &tmp);
            }
#endif
            return ret;
        }

        static std::string getOffset(long pos, std::string xml_file) {
            std::fstream is(xml_file);
            char buffer[1024];
            int line = 0, linestart = 0, offset = 0;
            while (is.good()) {
                is.read(buffer, sizeof(buffer));
                for (int i = 0; i < is.gcount(); ++i) {
                    if (buffer[i] == '\n') {
                        if (offset + i >= pos) {
                            std::stringstream ss;
                            std::string ret;
                            ss << "line " << line + 1 << ", col " << pos - linestart;
                            ss >> ret;
                            return ret;
                        }
                        ++line;
                        linestart = offset + i;
                    }
                }
                offset += (int) is.gcount();
            }
            return "byte offset " + std::to_string(pos);
        }

        static void handleTagBoolean(pugi::xml_node &node, XmlParseInfo &parentParseInfo) {
            std::string name = node.attribute("name").value();
            std::string value = node.attribute("value").value();
            ASSERT(value == "true" || value == "false", "Can't convert " + value + " to bool type");
            parentParseInfo.setBoolValue(name, value == "true" ? true : false);
        }

        static void handleTagInteger(pugi::xml_node &node, XmlParseInfo &parentParseInfo) {
            std::string name = node.attribute("name").value();
            std::string value = node.attribute("value").value();

            char *tmp = nullptr;
            int ret = strtol(value.c_str(), &tmp, 10);
            ASSERT(*tmp == '\0', "Can't convert " + value + " to int type");

            parentParseInfo.setIntValue(name, ret);
        }

        static inline Float toFloat(const std::string &val) {
            char *tmp = nullptr;
#if defined(_RENDER_DATA_DOUBLE_)
            Float v = strtod(val.c_str(), &tmp);
#else
            Float v = strtof(val.c_str(), &tmp);
#endif
            ASSERT(*tmp == '\0', "Can't convert " + val + " to float type");
            return v;
        }

        static void handleTagString(pugi::xml_node &node, XmlParseInfo &parentParseInfo) {
            std::string name = node.attribute("name").value();
            std::string value = node.attribute("value").value();
            parentParseInfo.setStringValue(name, value);
        }

        static void handleTagFloat(pugi::xml_node &node, XmlParseInfo &parentParseInfo) {
            std::string name = node.attribute("name").value();
            std::string value = node.attribute("value").value();

            Float ret = toFloat(value);
            parentParseInfo.setFloatValue(name, ret);
        }

        static void handleTagMatrix(pugi::xml_node &node, XmlParseInfo &parentParseInfo) {
            std::string value = node.attribute("value").value();

            Matrix4F ret;
            char *tmp = nullptr;
#if defined(_RENDER_DATA_DOUBLE_)
            ret[0][0] = strtod(value.c_str(), &tmp);
                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 4;j ++) {
                        if (i == 0 && j == 0) {
                            continue;
                        }
                        tmp++;
                        ret[j][i] = strtod(tmp, &tmp);
                    }
                }
#else
            ret[0][0] = strtof(value.c_str(), &tmp);
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    if (i == 0 && j == 0) {
                        continue;
                    }
                    tmp++;
                    ret[j][i] = strtof(tmp, &tmp);
                }
            }
#endif
            parentParseInfo.transformMat = Transform(ret);
        }

        static void handleTagTransform(pugi::xml_node &node, XmlParseInfo &parseInfo, XmlParseInfo &parentParseInfo) {
            std::string name = node.attribute("name").value();
            parentParseInfo.setTransformValue(name, parseInfo.transformMat);
        }

        static void handleTagFilm(pugi::xml_node &node, XmlParseInfo &parseInfo, SceneData &sceneData) {
            std::string type = node.attribute("type").value();
            sceneData.filmType = type;
            sceneData.filename = parseInfo.getStringValue("filename", sceneData.filename);
            sceneData.width = parseInfo.getIntValue("width", sceneData.width);
            sceneData.height = parseInfo.getIntValue("height", sceneData.height);
            sceneData.fileFormat = parseInfo.getStringValue("fileFormat", sceneData.fileFormat);
            sceneData.pixelFormat = parseInfo.getStringValue("pixelFormat", sceneData.pixelFormat);
            sceneData.gamma = parseInfo.getFloatValue("gamma", sceneData.gamma);
            sceneData.banner = parseInfo.getBoolValue("banner", sceneData.banner);
            sceneData.rfilter = parseInfo.getStringValue("rfilter", sceneData.rfilter);
        }

        static void handleTagSensor(pugi::xml_node &node, XmlParseInfo &parseInfo, SceneData &sceneData) {
            sceneData.type = node.attribute("type").value();
            ASSERT(sceneData.type == "perspective", "Only support perspective camera for now.");

            sceneData.fov = parseInfo.getFloatValue("fov", 45.);
            Transform toWorld = parseInfo.getTransformValue("toWorld", Transform());
            sceneData.cameraToWorld = toWorld;
        }

        static void handleTagIntegrator(pugi::xml_node &node, XmlParseInfo &info, SceneData &sceneData) {
            std::string type = node.attribute("type").value();
            sceneData.integratorType = type;
            sceneData.maxDepth = info.getIntValue("maxDepth", sceneData.maxDepth);
            sceneData.sampleNum = info.getIntValue("sampleNum", sceneData.sampleNum);
            sceneData.delta = info.getIntValue("delta", sceneData.delta);
            sceneData.strictNormals = info.getBoolValue("strictNormals", sceneData.strictNormals);
            sceneData.photonCount = info.getIntValue("photonCount", sceneData.photonCount);
            sceneData.initialRadius = info.getFloatValue("initialRadius", sceneData.initialRadius);
            sceneData.radiusDecay = info.getFloatValue("alpha", sceneData.radiusDecay);
        }

        void createRectangleShape(XmlParseInfo &info, SceneData &sceneData, ResourceManager *allocator) {
            sceneData.entities.push_back(ShapeEntity());
            ShapeEntity &entity = sceneData.entities.back();
            Transform toWorld = info.getTransformValue("toWorld", Transform());

            const int nVertices = 4;
            const int nNormals = nVertices;
            const int nTexcoords = nVertices;
            const int nTriangles = 2;

            // Vertices
            entity.nVertices = nVertices;
            entity.vertices = allocator->allocateObjects<Point3F>(nVertices);
            {
                entity.vertices[0] = toWorld.transformPoint(Point3F(-1, -1, 0));
                entity.vertices[1] = toWorld.transformPoint(Point3F(1, -1, 0));
                entity.vertices[2] = toWorld.transformPoint(Point3F(1, 1, 0));
                entity.vertices[3] = toWorld.transformPoint(Point3F(-1, 1, 0));
//                entity.vertices[0] = Point3F(-1, -1, 0);
//                entity.vertices[1] = Point3F(1, -1, 0);
//                entity.vertices[2] = Point3F(1, 1, 0);
//                entity.vertices[3] = Point3F(-1, 1, 0);
            }

            // Normals
            entity.nNormals = nNormals;
            entity.normals = allocator->allocateObjects<Normal3F>(nNormals);
            Normal3F normal(0, 0, 1);
            {
                entity.normals[0] = toWorld.transformNormal(normal);
                entity.normals[1] = toWorld.transformNormal(normal);
                entity.normals[2] = toWorld.transformNormal(normal);
                entity.normals[3] = toWorld.transformNormal(normal);
            }

            // UVs
            entity.nTexcoords = nTexcoords;
            entity.texcoords = allocator->allocateObjects<Point2F>(nTexcoords);
            {
                entity.texcoords[0] = Point2F(0, 0);
                entity.texcoords[1] = Point2F(1, 0);
                entity.texcoords[2] = Point2F(1, 1);
                entity.texcoords[3] = Point2F(0, 1);
            }

            // Indices
            entity.nTriangles = nTriangles;
            entity.vertexIndices = allocator->allocateObjects<int>(nTriangles * 3);
            entity.normalIndices = allocator->allocateObjects<int>(nTriangles * 3);
            entity.texcoordIndices = allocator->allocateObjects<int>(nTriangles * 3);
            {
                entity.vertexIndices[0] = 0;
                entity.normalIndices[0] = 0;
                entity.texcoordIndices[0] = 0;

                entity.vertexIndices[1] = 1;
                entity.normalIndices[1] = 1;
                entity.texcoordIndices[1] = 1;

                entity.vertexIndices[2] = 2;
                entity.normalIndices[2] = 2;
                entity.texcoordIndices[2] = 2;

                entity.vertexIndices[3] = 2;
                entity.normalIndices[3] = 2;
                entity.texcoordIndices[3] = 2;

                entity.vertexIndices[4] = 3;
                entity.normalIndices[4] = 3;
                entity.texcoordIndices[4] = 3;

                entity.vertexIndices[5] = 0;
                entity.normalIndices[5] = 0;
                entity.texcoordIndices[5] = 0;
            }
        }

        static void createCubeShape(XmlParseInfo &info, SceneData &sceneData, ResourceManager *allocator) {
            sceneData.entities.push_back(ShapeEntity());
            ShapeEntity &entity = sceneData.entities.back();
            Transform toWorld = info.getTransformValue("toWorld", Transform());

            const int nVertices = 8;
            const int nTriangles = 12;

            // Vertices
            entity.nVertices = nVertices;
            entity.vertices = allocator->allocateObjects<Point3F>(nVertices);
            {
                entity.vertices[0] = toWorld.transformPoint(Point3F(1, -1, -1));
                entity.vertices[1] = toWorld.transformPoint(Point3F(1, -1, 1));
                entity.vertices[2] = toWorld.transformPoint(Point3F(-1, -1, 1));
                entity.vertices[3] = toWorld.transformPoint(Point3F(-1, -1, -1));
                entity.vertices[4] = toWorld.transformPoint(Point3F(1, 1, -1));
                entity.vertices[5] = toWorld.transformPoint(Point3F(-1, 1, -1));
                entity.vertices[6] = toWorld.transformPoint(Point3F(-1, 1, 1));
                entity.vertices[7] = toWorld.transformPoint(Point3F(1, 1, 1));

//                entity.vertices[0] = Point3F(1, -1, -1);
//                entity.vertices[1] = Point3F(1, -1, 1);
//                entity.vertices[2] = Point3F(-1, -1, 1);
//                entity.vertices[3] = Point3F(-1, -1, -1);
//                entity.vertices[4] = Point3F(1, 1, -1);
//                entity.vertices[5] = Point3F(-1, 1, -1);
//                entity.vertices[6] = Point3F(-1, 1, 1);
//                entity.vertices[7] = Point3F(1, 1, 1);
            }

            // Vertex indices
            entity.nTriangles = nTriangles;
            entity.vertexIndices = allocator->allocateObjects<int>(nTriangles * 3);
            {
                entity.vertexIndices[0] = 7;
                entity.vertexIndices[1] = 2;
                entity.vertexIndices[2] = 1;

                entity.vertexIndices[3] = 7;
                entity.vertexIndices[4] = 6;
                entity.vertexIndices[5] = 2;

                entity.vertexIndices[6] = 4;
                entity.vertexIndices[7] = 1;
                entity.vertexIndices[8] = 0;

                entity.vertexIndices[9] = 4;
                entity.vertexIndices[10] = 7;
                entity.vertexIndices[11] = 1;

                entity.vertexIndices[12] = 5;
                entity.vertexIndices[13] = 0;
                entity.vertexIndices[14] = 3;

                entity.vertexIndices[15] = 5;
                entity.vertexIndices[16] = 4;
                entity.vertexIndices[17] = 0;

                entity.vertexIndices[18] = 6;
                entity.vertexIndices[19] = 3;
                entity.vertexIndices[20] = 2;

                entity.vertexIndices[21] = 6;
                entity.vertexIndices[22] = 5;
                entity.vertexIndices[23] = 3;

                entity.vertexIndices[24] = 4;
                entity.vertexIndices[25] = 6;
                entity.vertexIndices[26] = 7;

                entity.vertexIndices[27] = 4;
                entity.vertexIndices[28] = 5;
                entity.vertexIndices[29] = 6;

                entity.vertexIndices[30] = 1;
                entity.vertexIndices[31] = 2;
                entity.vertexIndices[32] = 3;

                entity.vertexIndices[33] = 1;
                entity.vertexIndices[34] = 3;
                entity.vertexIndices[35] = 0;
            }

            // Normals and normal indices
            entity.nNormals = nTriangles * 3;
            entity.normals = allocator->allocateObjects<Normal3F>(nTriangles * 3);
            entity.normalIndices = allocator->allocateObjects<int>(nTriangles * 3);
            for (int i = 0; i < nTriangles; i++) {
                int index = i * 3;
                Point3F v1 = entity.vertices[entity.vertexIndices[index]];
                Point3F v2 = entity.vertices[entity.vertexIndices[index + 1]];
                Point3F v3 = entity.vertices[entity.vertexIndices[index + 2]];

                Normal3F normal = NORMALIZE(CROSS(v2 - v1, v3 - v1));
                for (int j = 0; j < 3; j++) {
                    entity.normals[index + j] = normal;
                    entity.normalIndices[index + j] = index + j;
                }
            }

            // UVs
            entity.nTexcoords = 1;
            entity.texcoords = allocator->allocateObjects<Point2F>(1);
            entity.texcoordIndices = allocator->allocateObjects<int>(nTriangles * 3);
            {
                entity.texcoords[0] = Point2F(0.0f, 0.0f);
                for (int i = 0; i < nTriangles; i++) {
                    int index = i * 3;
                    for (int j = 0; j < 3; j++) {
                        entity.texcoordIndices[index + j] = 0;
                    }
                }
            }
        }

        static void createObjMeshes(XmlParseInfo &info, SceneData &sceneData, ResourceManager *allocator) {
            sceneData.entities.push_back(ShapeEntity());
            ShapeEntity &entity = sceneData.entities.back();
            Transform toWorld = info.getTransformValue("toWorld", Transform());

            // face normal
            bool faceNormal = false;
            faceNormal = info.getBoolValue("faceNormals", false);
            entity.faceNormal = faceNormal;

            std::string filename = info.getStringValue("filename", "");
            ASSERT(filename != "", "Obj filename can't be empty. ");

            bool good = utils::load_obj(sceneData.sceneDirectory + filename, entity, allocator);
            ASSERT(good, "Load *.obj model failed: " + filename);

            // Transform mesh from object space to world space
            {
                for (int i = 0; i < entity.nVertices; i++) {
                    entity.vertices[i] = toWorld.transformPoint(entity.vertices[i]);
                }

                for (int i = 0; i < entity.nNormals; i++) {
                    entity.normals[i] = toWorld.transformNormal(entity.normals[i]);
                }
            }

            std::cout << "\tLoading mesh: " << filename << std::endl;
            return;
        }

        Material createDiffuseMaterial(XmlParseInfo &info, ResourceManager *allocator) {
            Material material;
            if (!info.attrExists("reflectance")) {
                // Create default diffuse material
                SpectrumTexture texture = allocator->newObject<ConstantSpectrumTexture>(Spectrum(0.f));
                material = allocator->newObject<Lambertian>(texture);
            } else {
                auto type = info.getType("reflectance");
                if (type == XmlAttrVal::Attr_Spectrum) {
                    Spectrum Kd = info.getSpectrumValue("reflectance", Spectrum(0));
                    SpectrumTexture texture = allocator->newObject<ConstantSpectrumTexture>(Spectrum(Kd));
                    material = allocator->newObject<Lambertian>(texture);
                } else if (type == XmlAttrVal::Attr_SpectrumTexture) {
                    SpectrumTexture texture = info.getSpectrumTextureValue("reflectance", SpectrumTexture());
                    ASSERT(!texture.nullable(), "Texture can't be nullptr. ");
                    material = allocator->newObject<Lambertian>(texture);
                } else {
                    // TODO
                    ASSERT(false, "Reflectance type not supported .");
                }
            }
            return material;
        }

        Material createDielectricMaterial(XmlParseInfo &info, ResourceManager *allocator) {
            Material material;
            auto intIORType = info.getType("intIOR");
            auto extIORType = info.getType("extIOR");
            ASSERT(intIORType == XmlAttrVal::Attr_Float && extIORType == XmlAttrVal::Attr_Float,
                   "Only support float type IOR for now");

            Float roughness = info.getFloatValue("alpha", 0.0);
            // thetaT
            Float intIOR = info.getFloatValue("intIOR", 1.5);
            // thetaI
            Float extIOR = info.getFloatValue("extIOR", 1.0);
            SpectrumTexture texR = allocator->newObject<ConstantSpectrumTexture>(Spectrum(1.0f));
            SpectrumTexture texT = allocator->newObject<ConstantSpectrumTexture>(Spectrum(1.0f));
            material = allocator->newObject<Dielectric>(texR, texT, extIOR, intIOR, roughness);
            return material;
        }

        Material createMirrorMaterial(XmlParseInfo &info, ResourceManager *allocator) {
            return allocator->newObject<Mirror>();
        }

        static Material createGlassMaterial(XmlParseInfo &info, ResourceManager *allocator) {
            Float extIOR = 1.;
            Float intIOR = 1.5;
            SpectrumTexture texR = allocator->newObject<ConstantSpectrumTexture>(Spectrum(1.0f));
            SpectrumTexture texT = allocator->newObject<ConstantSpectrumTexture>(Spectrum(1.0f));
            return allocator->newObject<Dielectric>(texR, texT, extIOR, intIOR);
        }

        static Material createRoughConductorMaterial(XmlParseInfo &info, ResourceManager *allocator) {
            // Roughness
            Float alpha = 0.01;
            if (info.attrExists("alpha")) {
                auto alphaType = info.getType("alpha");
                ASSERT(alphaType == XmlAttrVal::Attr_Float, "Only support float type for alpha. ")
                alpha = info.getFloatValue("alpha", 0.01);
            } else {
                alpha = info.getFloatValue("alpha", 0.01);
            }

            std::string distributionTypeString = info.getStringValue("distribution", "ggx");
            MicrofacetDistribType distribType = GGX;
            if (distributionTypeString == "ggx") {
                distribType = GGX;
            } else {
                ASSERT(false, "Microfacet distribution unsupported: " + distributionTypeString);
            }

            Spectrum specularReflectance = info.getSpectrumValue("specularReflectance", Spectrum(1.0));
            Spectrum eta = info.getSpectrumValue("eta", Spectrum(0.200438));
            Spectrum k = info.getSpectrumValue("k", Spectrum(0.200438));

            FloatTexture Alpha = allocator->newObject<ConstantFloatTexture>(alpha);
            SpectrumTexture Ks = allocator->newObject<ConstantSpectrumTexture>(specularReflectance);
            SpectrumTexture Eta = allocator->newObject<ConstantSpectrumTexture>(eta);
            SpectrumTexture K = allocator->newObject<ConstantSpectrumTexture>(k);
            return allocator->newObject<Metal>(Alpha, Eta, Ks, K, distribType);
        }

        // Rename method
        static Material createCoatingMaterial(XmlParseInfo &info, ResourceManager *allocator) {
            Material material;
            ASSERT(info.attrExists("diffuseReflectance") &&
                   info.attrExists("specularReflectance") && info.attrExists("alpha"),
                   "CoatingMaterial parameter error: type not supported");

            // Kd
            SpectrumTexture Kd;
            XmlAttrVal::AttrType kdType = info.getType("diffuseReflectance");
            if (kdType == XmlAttrVal::Attr_Spectrum) {
                Spectrum diffuseReflectance = info.getSpectrumValue("diffuseReflectance", Spectrum(1.0));
                Kd = allocator->newObject<ConstantSpectrumTexture>(diffuseReflectance);
            } else if (kdType == XmlAttrVal::Attr_SpectrumTexture) {
                Kd = info.getSpectrumTextureValue("diffuseReflectance", SpectrumTexture());
            } else {
                ASSERT(false, "Unsupported Kd type. ");
            }

            // Ks
            SpectrumTexture Ks;
            XmlAttrVal::AttrType ksType = info.getType("specularReflectance");
            if (ksType == XmlAttrVal::Attr_Spectrum) {
                Spectrum specularReflectance = info.getSpectrumValue("specularReflectance", Spectrum(1.0));
                Ks = allocator->newObject<ConstantSpectrumTexture>(specularReflectance);
            } else if (ksType == XmlAttrVal::Attr_SpectrumTexture) {
                Ks = info.getSpectrumTextureValue("specularReflectance", SpectrumTexture());
            } else {
                ASSERT(false, "Unsupported Ks type. ");
            }

            auto alpha = info.getFloatValue("alpha", 0.1);

            FloatTexture roughness = allocator->newObject<ConstantFloatTexture>(alpha);
            material = allocator->newObject<Patina>(Kd, Ks, roughness, GGX);
            return material;
        }

        void handleTagBSDF(pugi::xml_node &node, XmlParseInfo &parseInfo, XmlParseInfo &parent,
                           SceneData &sceneData, ResourceManager *allocator) {
            std::string type = node.attribute("type").value();
            std::string id = node.attribute("id").value();

            Material material;
            if (type == "diffuse") {
                material = createDiffuseMaterial(parseInfo, allocator);
            } else if (type == "dielectric") {
                material = createDielectricMaterial(parseInfo, allocator);
            } else if (type == "mirror") {
                material = createMirrorMaterial(parseInfo, allocator);
            } else if (type == "glass") {
                material = createGlassMaterial(parseInfo, allocator);
            } else if (type == "roughconductor" || type == "conductor") {
                material = createRoughConductorMaterial(parseInfo, allocator);
            } else if (type == "twosided") {
                material = parseInfo.currentMaterial;
            } else if (type == "coating") {
                material = createCoatingMaterial(parseInfo, allocator);
                /*
            } else if (type == "plastic") {
                material = createPlasticMaterial(parseInfo);
                 */
            } else {
                ASSERT(false, "Material " + type + " not supported for now");
            }

            if (id == "") {
                parent.currentMaterial = (const Material) (material);
            } else {
                sceneData.materialMap[id] = (const Material) (material);
            }
        }

        static void handleTagShape(pugi::xml_node &node, XmlParseInfo &parseInfo,
                                   SceneData &sceneData, ResourceManager *allocator) {
            std::string type = node.attribute("type").value();

            if (type == "rectangle") {
                createRectangleShape(parseInfo, sceneData, allocator);
            } else if (type == "cube") {
                createCubeShape(parseInfo, sceneData, allocator);
            } else if (type == "obj") {
                createObjMeshes(parseInfo, sceneData, allocator);
            } else {
                ASSERT(false, "Only support rectangle shape for now");
            }

            Material material = parseInfo.currentMaterial;
            Medium exteriorMedium = parseInfo.currentExteriorMedium;
            Medium interiorMedium = parseInfo.currentInteriorMedium;

            ShapeEntity &entity = sceneData.entities.back();
            entity.material = material;
            entity.interiorMedium = interiorMedium;
            entity.exteriorMedium = exteriorMedium;

            Spectrum radiance(0.0);
            if (parseInfo.hasAreaLight) {
                auto radianceType = parseInfo.getType("radiance");
                if (radianceType == XmlAttrVal::Attr_Spectrum) {
                    radiance = parseInfo.getSpectrumValue("radiance", Spectrum(0));
                } else {
                    ASSERT(false, "Only support spectrum radiance for now.");
                }
                entity.createAreaLights(radiance, allocator);
                for (int i = 0; i < entity.nTriangles; i++) {
                    sceneData.lights->push_back(entity.areaLights + i);
                }
            }
        }

        static void handleTagLookAt(pugi::xml_node &node, XmlParseInfo &parent) {
            const Vector3F origin = toVector(node.attribute("origin").value());
            const Vector3F target = toVector(node.attribute("target").value());
            const Vector3F up = toVector(node.attribute("up").value());

            Matrix4F mat;
            mat[3][0] = origin[0];
            mat[3][1] = origin[1];
            mat[3][2] = origin[2];
            mat[3][3] = 1;

            Vector3F forward = NORMALIZE(target - origin);
            Vector3F left = CROSS(up, forward);
            Vector3F realUp = CROSS(forward, left);
            mat[0][0] = left[0];
            mat[0][1] = left[1];
            mat[0][2] = left[2];
            mat[0][3] = 0;

            mat[1][0] = realUp[0];
            mat[1][1] = realUp[1];
            mat[1][2] = realUp[2];
            mat[1][3] = 0;

            mat[2][0] = forward[0];
            mat[2][1] = forward[1];
            mat[2][2] = forward[2];
            mat[2][3] = 0;

            parent.transformMat = Transform(mat);
        }

        static void handleTagRef(pugi::xml_node &node, XmlParseInfo &parent, SceneData &sceneData) {
            std::string id = node.attribute("id").value();
            ASSERT(sceneData.materialMap.count(id) > 0, "Material " + id + " Not Exists.!");
            parent.currentMaterial = (const Material) (sceneData.materialMap[id]);
        }

        void handleTagRGB(pugi::xml_node &node, XmlParseInfo &parentParseInfo) {
            if (strcmp(node.name(), "spectrum") == 0) {
                // TODO: Fix Spectrum declared with wavelength
                ASSERT(false, "No implemented!");
            } else if (strcmp(node.name(), "rgb") == 0 || strcmp(node.name(), "color") == 0) {
                Spectrum ret;
                std::string colorValue = node.attribute("value").value();
                char *tmp;
#if defined(_RENDER_DATA_DOUBLE_)
                ret[0] = strtod(colorValue.c_str(), &tmp);
                    for (int i = 1; i < 3; i++) {
                        tmp++;
                        ret[i] = strtod(tmp, &tmp);
                    }
#else
                ret[0] = strtof(colorValue.c_str(), &tmp);
                for (int i = 1; i < 3; i++) {
                    tmp++;
                    ret[i] = strtof(tmp, &tmp);
                }
#endif
                std::string name = node.attribute("name").value();
                parentParseInfo.setSpectrumValue(name, ret);
            }
        }

        static void
        handleTagEmitter(pugi::xml_node &node, XmlParseInfo &info, XmlParseInfo &parent,
                         SceneData &sceneData, ResourceManager *allocator) {
            std::string type = node.attribute("type").value();
            if (type == "area") {
                parent.hasAreaLight = true;
                auto radianceType = info.getType("radiance");
                if (radianceType == XmlAttrVal::Attr_Spectrum) {
                    parent.set("radiance", info.get("radiance"));
                } else {
                    ASSERT(false, "Only support spectrum type radiance.");
                }
                std::cout << "\tCreate light: area light" << std::endl;
            } else if (type == "point") {
                Transform toWorld = info.getTransformValue("toWorld", Transform());
                Spectrum intensity = info.getSpectrumValue("intensity", Spectrum(0));
                Light pointLight = allocator->newObject<PointLight>(intensity, toWorld, MediumInterface());
                sceneData.lights->push_back(pointLight);
                std::cout << "\tCreate light: point light" << std::endl;
            } else if (type == "spot") {
                Transform toWorld = info.getTransformValue("toWorld", Transform());
                Spectrum intensity = info.getSpectrumValue("intensity", Spectrum(0));
                Float totalAngle = info.getFloatValue("totalAngle", 60.);
                Float falloffAngle = info.getFloatValue("falloffAngle", 50.);
                Light spotLight = allocator->newObject<SpotLight>(intensity, toWorld, MediumInterface(),
                                                                  falloffAngle, totalAngle);
                sceneData.lights->push_back(spotLight);
                std::cout << "\tCreate light: spot light" << std::endl;
            } else if (type == "envmap") {
                Transform toWorld = info.getTransformValue("toWorld", Transform());

                ASSERT(info.attrExists("filename"), "Environment light type should has envmap. ");
                std::string envmapPath = info.getStringValue("filename", "");

                EnvironmentLight *envLight = allocator->newObject<EnvironmentLight>(1., sceneData.sceneDirectory +
                                                                                        envmapPath, MediumInterface(),
                                                                                    toWorld, allocator);
                sceneData.lights->push_back(envLight);
                sceneData.envLights->push_back(envLight);
                std::cout << "\tCreate environment light. " << std::endl;
                /*
            } else if (type == "sunsky") {
                ASSERT(info.attrExists("sunDirection") && info.attrExists("intensity"),
                       "Sunsky parameter incomplete. ");
                ASSERT(info.getType("intensity") == XmlAttrVal::Attr_Spectrum,
                       "<emitter> Only support spectrum intensity. ");
                Spectrum intensity = info.getSpectrumValue("intensity", Spectrum(0.0));
                Vector3F sunDirection = -info.getVectorValue("sunDirection", Vector3F(0, 1, 0));
                SunLight::Ptr sunLight = std::make_shared<SunLight>(intensity, sunDirection);
                _scene->addLight(sunLight);
                std::cout << "\tCreate sun light. " << std::endl;
                 */
            } else {
                ASSERT(false, "Emitter type not supported: <" + type + ">. ");
            }
        }

        static void handleTagTexture(pugi::xml_node &node, XmlParseInfo &parseInfo, XmlParseInfo &parentInfo,
                                     SceneData &sceneData, ResourceManager *allocator) {
            const std::string type = node.attribute("type").value();
            const std::string name = node.attribute("name").value();
            if (type == "checkerboard") {
                Spectrum color0 = parseInfo.getSpectrumValue("color0", Spectrum(0.));
                Spectrum color1 = parseInfo.getSpectrumValue("color1", Spectrum(0.));

                Float uScale = parseInfo.getFloatValue("uscale", 1.0);
                Float vScale = parseInfo.getFloatValue("vscale", 1.0);

                SpectrumTexture texture = allocator->newObject<ChessboardSpectrumTexture>(color0, color1,
                                                                                          uScale, vScale);
                parentInfo.setSpectrumTextureValue(name, texture);
            } else if (type == "bitmap") {
                ASSERT(parseInfo.attrExists("filename"), "Bitmap path not exists. ");
                std::string filename = parseInfo.getStringValue("filename", "");
                TextureMapping2D mapping = allocator->newObject<UVMapping2D>();
                SpectrumTexture texture = allocator->newObject<ImageSpectrumTexture>(sceneData.sceneDirectory +
                                                                                     filename, mapping, allocator);

                parentInfo.setSpectrumTextureValue(name, texture);
            } else {
                ASSERT(false, "Texture type not supported: " + type);
            }
        }

        static void handleTagMedium(pugi::xml_node &node, XmlParseInfo &info, XmlParseInfo &parent,
                                    ResourceManager *allocator) {
            std::string mediumType = node.attribute("type").value();
            std::string mediumName = node.attribute("name").value();
            Medium medium;
            if (mediumType == "homogeneous") {
                ASSERT(info.attrExists("sigmaS") && info.attrExists("sigmaA"), "Medium parameter missed. ")
                Spectrum sigmaS = info.getSpectrumValue("sigmaS", Spectrum(0.1));
                Spectrum sigmaA = info.getSpectrumValue("sigmaA", Spectrum(0.1));
                Float g = info.getFloatValue("g", 0.0f);
                medium = allocator->newObject<HomogenousMedium>(sigmaA, sigmaS, g);
            } else {
                ASSERT(false, "Medium type not supported. ");
            }

            if (!medium.nullable()) {
                if (mediumName == "exterior") {
                    parent.currentExteriorMedium = medium;
                } else if (mediumName == "interior") {
                    parent.currentInteriorMedium = medium;
                }
            }
        }

        static void handleXmlNode(pugi::xml_node &node, XmlParseInfo &parseInfo, XmlParseInfo &parentParseInfo,
                                  SceneData &sceneData, ResourceManager *allocator) {
            TagType tagType = nodeTypeMap[node.name()];
            switch (tagType) {
//                case Tag_Mode:
//                    handleTagMode(node, parseInfo);
//                    break;
                case Tag_Boolean:
                    handleTagBoolean(node, parentParseInfo);
                    break;
                case Tag_Integer:
                    handleTagInteger(node, parentParseInfo);
                    break;
                case Tag_Float:
                    handleTagFloat(node, parentParseInfo);
                    break;
                case Tag_Matrix:
                    handleTagMatrix(node, parentParseInfo);
                    break;
//                case Tag_Vector:
//                    handleTagVector(node, parentParseInfo);
//                    break;
                case Tag_Transform:
                    handleTagTransform(node, parseInfo, parentParseInfo);
                    break;
                case Tag_String:
                    handleTagString(node, parentParseInfo);
                    break;
                    /*
                case Tag_Sampler:
                    handleTagSampler(node, parseInfo);
                    break;
                     */
                case Tag_Film:
                    handleTagFilm(node, parseInfo, sceneData);
                    break;
                case Tag_Sensor:
                    handleTagSensor(node, parseInfo, sceneData);
                    break;
                case Tag_Texture:
                    handleTagTexture(node, parseInfo, parentParseInfo, sceneData, allocator);
                    break;
                case Tag_BSDF:
                    handleTagBSDF(node, parseInfo, parentParseInfo, sceneData, allocator);
                    break;
                case Tag_Shape:
                    handleTagShape(node, parseInfo, sceneData, allocator);
                    break;
                case Tag_Ref:
                    handleTagRef(node, parentParseInfo, sceneData);
                    break;
                case Tag_RGB:
                    handleTagRGB(node, parentParseInfo);
                    break;
                case Tag_Emitter:
                    handleTagEmitter(node, parseInfo, parentParseInfo, sceneData, allocator);
                    break;
                case Tag_LookAt:
                    handleTagLookAt(node, parentParseInfo);
                    break;
                case Tag_Integrator:
                    handleTagIntegrator(node, parseInfo, sceneData);
                    break;
                case Tag_Medium:
                    handleTagMedium(node, parseInfo, parentParseInfo, allocator);
                    break;
                default:
                    std::cout << "\tUnsupported tag: <" << node.name() << ">" << std::endl;
            }
        }

        void parseXml(pugi::xml_node &node, XmlParseInfo &parent, SceneData &scene, ResourceManager *allocator) {
            XmlParseInfo info;
            std::map <std::string, XmlAttrVal> attrContainer;
            for (pugi::xml_node &child : node.children()) {
                parseXml(child, info, scene, allocator);
            }
            handleXmlNode(node, info, parent, scene, allocator);
        }

        void MitsubaSceneImporter::importScene(std::string sceneDirectory, SceneData &sceneData,
                                               ResourceManager *allocator) {
            sceneData.sceneDirectory = sceneDirectory;
            sceneData.lights = allocator->newObject < base::Vector < Light >> (allocator);
            sceneData.envLights = allocator->newObject < base::Vector < EnvironmentLight * >> (allocator);

            std::string xml_file = sceneDirectory + "scene.xml";
            std::cout << "Loading scene file: " << xml_file << std::endl;

            pugi::xml_document xml_doc;
            pugi::xml_parse_result ret = xml_doc.load_file(xml_file.c_str());

            ASSERT(ret, "Error while parsing \"" + xml_file + "\": " + ret.description()
                        + " (at " + getOffset(ret.offset, xml_file) + ")");

            XmlParseInfo parseInfo;
            parseXml(*xml_doc.begin(), parseInfo, sceneData, allocator);
            std::cout << "\tReading scene data finished ." << std::endl;

            sceneData.postprocess();
            std::cout << "Loading finished. " << std::endl;
        }

        MitsubaSceneImporter::MitsubaSceneImporter() {
            if (nodeTypeMap.empty()) {
                nodeTypeMap["mode"] = Tag_Mode;
                nodeTypeMap["scene"] = Tag_Scene;

                nodeTypeMap["integrator"] = Tag_Integrator;
                nodeTypeMap["sensor"] = Tag_Sensor;
                nodeTypeMap["camera"] = Tag_Sensor;
                nodeTypeMap["sampler"] = Tag_Sampler;
                nodeTypeMap["film"] = Tag_Film;
                nodeTypeMap["rfilter"] = Tag_Rfilter;
                nodeTypeMap["emitter"] = Tag_Emitter;
                nodeTypeMap["shape"] = Tag_Shape;
                nodeTypeMap["bsdf"] = Tag_BSDF;
                nodeTypeMap["ref"] = Tag_Ref;

                nodeTypeMap["texture"] = Tag_Texture;
                nodeTypeMap["medium"] = Tag_Medium;
                nodeTypeMap["bool"] = Tag_Boolean;
                nodeTypeMap["boolean"] = Tag_Boolean;
                nodeTypeMap["integer"] = Tag_Integer;
                nodeTypeMap["float"] = Tag_Float;
                nodeTypeMap["string"] = Tag_String;
                nodeTypeMap["rgb"] = Tag_RGB;
                nodeTypeMap["transform"] = Tag_Transform;
                nodeTypeMap["matrix"] = Tag_Matrix;
                nodeTypeMap["vector"] = Tag_Vector;
                nodeTypeMap["lookat"] = Tag_LookAt;
            }
        }
    }
}