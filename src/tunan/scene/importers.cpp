//
// Created by StormPhoenix on 2021/6/2.
//

#include <tunan/scene/importers.h>
#include <tunan/base/transform.h>
#include <tunan/base/spectrum.h>

#include <ext/pugixml/pugixml.hpp>

#include <fstream>
#include <sstream>

namespace RENDER_NAMESPACE {
    namespace importer {
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

            GET_PARSE_INFO_VALUE_FUNC_DECLARE(float, Float)

            SET_PARSE_INFO_VALUE_FUNC_DECLARE(float, Float)

            GET_PARSE_INFO_VALUE_FUNC_DECLARE(Transform, Transform)

            SET_PARSE_INFO_VALUE_FUNC_DECLARE(Transform, Transform)

            GET_PARSE_INFO_VALUE_FUNC_DECLARE(std::string, String)

            SET_PARSE_INFO_VALUE_FUNC_DECLARE(std::string, String)

            SET_PARSE_INFO_VALUE_FUNC_DECLARE(Vector3F, Vector)

            GET_PARSE_INFO_VALUE_FUNC_DECLARE(Vector3F, Vector)

        private:
            std::map<std::string, XmlAttrVal> container;
        } XmlParseInfo;

        GET_PARSE_INFO_VALUE_FUNC_DEFINE(bool, Bool, bool);

        SET_PARSE_INFO_VALUE_FUNC_DEFINE(bool, Bool, bool);

        GET_PARSE_INFO_VALUE_FUNC_DEFINE(int, Int, int);

        SET_PARSE_INFO_VALUE_FUNC_DEFINE(int, Int, int);

        GET_PARSE_INFO_VALUE_FUNC_DEFINE(Spectrum, Spectrum, spectrum);

        SET_PARSE_INFO_VALUE_FUNC_DEFINE(Spectrum, Spectrum, spectrum);

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
        };

        static std::map<std::string, TagType> nodeTypeMap;

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

        static void
        handleXmlNode(pugi::xml_node &node, XmlParseInfo &parseInfo,
                      XmlParseInfo &parentParseInfo, SceneData &sceneData) {
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
//                case Tag_Texture:
//                    handleTagTexture(node, parseInfo, parentParseInfo);
//                    break;
//                case Tag_BSDF:
//                    handleTagBSDF(node, parseInfo, parentParseInfo);
//                    break;
                case Tag_Shape:
                    handleTagShape(node, parseInfo);
                    break;
//                case Tag_Ref:
//                    handleTagRef(node, parentParseInfo);
//                    break;
//                case Tag_RGB:
//                    handleTagRGB(node, parentParseInfo);
//                    break;
//                case Tag_Emitter:
//                    handleTagEmitter(node, parseInfo, parentParseInfo);
//                    break;
//                case Tag_LookAt:
//                    handleTagLookAt(node, parentParseInfo);
//                    break;
                case Tag_Integrator:
                    handleTagIntegrator(node, parseInfo, sceneData);
                    break;
//                case Tag_Medium:
//                    handleTagMedium(node, parseInfo, parentParseInfo);
//                    break;
                default:
                    std::cout << "\tUnsupported tag: <" << node.name() << ">" << std::endl;
            }
        }

        void parseXml(pugi::xml_node &node, XmlParseInfo &parent, SceneData &scene) {
            XmlParseInfo info;
            std::map<std::string, XmlAttrVal> attrContainer;
            for (pugi::xml_node &child : node.children()) {
                parseXml(child, info, scene);
            }
            handleXmlNode(node, info, parent, scene);
        }

        void MitsubaSceneImporter::importScene(std::string sceneDir, SceneData &scene, MemoryAllocator &allocator) {
            _inputSceneDir = sceneDir;
            std::string xml_file = sceneDir + "scene.xml";
            std::cout << "Loading scene file: " << xml_file << std::endl;

            pugi::xml_document xml_doc;
            pugi::xml_parse_result ret = xml_doc.load_file(xml_file.c_str());

            ASSERT(ret, "Error while parsing \"" + xml_file + "\": " + ret.description()
                        + " (at " + getOffset(ret.offset, xml_file) + ")");

            XmlParseInfo parseInfo;
            parseXml(*xml_doc.begin(), parseInfo, scene);

            std::cout << "\tBuilding bvh acceleration structure ... " << std::endl;
            std::shared_ptr<Intersectable> bvh = std::make_shared<BVH>(_shapes);
            _scene->setWorld(bvh);
            _scene->setSceneName(Config::Camera::filename);

            const std::vector<Light::Ptr> &lights = _scene->getLights();
            for (auto it = lights.begin(); it != lights.end(); it++) {
                (*it)->worldBound(_scene);
            }

            std::cout << "Loading finished. " << std::endl;
            return _scene;
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