//
// Created by Storm Phoenix on 2021/6/8.
//

#ifndef TUNAN_LIGHTS_H
#define TUNAN_LIGHTS_H

#include <tunan/common.h>
#include <tunan/scene/shapes.h>
#include <tunan/base/mediums.h>
#include <tunan/base/spectrum.h>
#include <tunan/base/transform.h>
#include <tunan/base/interactions.h>
#include <tunan/base/distributions.h>
#include <tunan/utils/TaggedPointer.h>

namespace RENDER_NAMESPACE {
    using namespace base;
    namespace utils {
        class ResourceManager;
    }

    typedef enum LightSourceType {
        Delta_Position = 1 << 0,
        Delta_Direction = 1 << 1,
        Area = 1 << 2,
        Environment = 1 << 3
    } LightSourceType;

    class EnvironmentLight {
    public:
        EnvironmentLight(Float intensity, std::string texturePath, MediumInterface mediumInterface,
                         Transform lightToWorld, utils::ResourceManager *allocator);

        RENDER_CPU_GPU
        Spectrum sampleLi(const Interaction &eye, Vector3F *wi, Float *pdf, Vector2F uv, Interaction *target);

        RENDER_CPU_GPU
        Float pdfLi(const Interaction &eye, const Vector3F &dir);

        RENDER_CPU_GPU
        Spectrum Le(const Ray &ray) const;

        RENDER_CPU_GPU
        void worldBound(Point3F &worldMin, Point3F &worldMax);

        RENDER_CPU_GPU
        LightSourceType getType() const;
    private:
        RENDER_CPU_GPU
        Spectrum sampleTexture(Point2F uv) const;

    private:
        LightSourceType _type;
        Float _intensity = 12;
        MediumInterface _mediumInterface;

        Transform _lightToWorld;
        Transform _worldToLight;
        Float _worldRadius = 20000;
        Point3F _worldCenter;

        // Texture
        Spectrum *_texture = nullptr;
        int _width, _height;

        // Texture distribution
        Distribution2D _textureDistribution;
    };

    class SpotLight {
    public:
        SpotLight(const Spectrum &intensity, Transform lightToWorld, MediumInterface mediumBoundary,
                  Float fallOffRange = 30, Float totalRange = 45);

        RENDER_CPU_GPU
        Spectrum sampleLi(const Interaction &eye, Vector3F *wi, Float *pdf,
                          Vector2F uv, Interaction *target);

        RENDER_CPU_GPU
        Float pdfLi(const Interaction &eye, const Vector3F &direction);

        RENDER_CPU_GPU
        Spectrum Le(const Ray &ray) const {
            return Spectrum(0.f);
        }

        RENDER_CPU_GPU
        inline LightSourceType getType() const {
            return _type;
        }

    private:
        RENDER_CPU_GPU
        Spectrum fallOffWeight(const Vector3F &wo);

    private:
        LightSourceType _type;
        Float _cosFallOffRange;
        Float _cosTotalRange;
        Spectrum _intensity;
        Vector3F _center, _direction;
        Transform _lightToWorld;
        MediumInterface _mediumInterface;
    };

    class PointLight {
    public:
        PointLight(const Spectrum &intensity, Transform lightToWorld, MediumInterface mediumInterface);

        RENDER_CPU_GPU
        Spectrum sampleLi(const Interaction &eye, Vector3F *wi, Float *pdf, Vector2F uv, Interaction *target);

        RENDER_CPU_GPU
        Float pdfLi(const Interaction &eye, const Vector3F &direction);

        RENDER_CPU_GPU
        Spectrum Le(const Ray &ray) const {
            return Spectrum(0.f);
        }

        RENDER_CPU_GPU
        inline LightSourceType getType() const {
            return _type;
        }

    private:
        LightSourceType _type;
        Vector3F _center;
        Spectrum _intensity;
        Transform _lightToWorld;
        MediumInterface _mediumInterface;
    };

    class DiffuseAreaLight {
    public:
        DiffuseAreaLight(const Spectrum &radiance, Shape shape,
                         MediumInterface mediumInterface, bool twoSided = false);

        RENDER_CPU_GPU
        Spectrum L(const Interaction &interaction, const Vector3F &wo) const;

        RENDER_CPU_GPU
        Float pdfLi(const Interaction &eye, const Vector3F &dir);

        RENDER_CPU_GPU
        Spectrum sampleLi(const Interaction &eye, Vector3F *wi, Float *pdf, Vector2F uv, Interaction *target);

        RENDER_CPU_GPU
        Spectrum Le(const Ray &ray) const {
            return Spectrum(0.f);
        }

        RENDER_CPU_GPU
        LightSourceType getType() const;

    protected:
        Shape _shape;
        bool _twoSided = false;
        Spectrum _radiance;
        LightSourceType _type = Area;
        MediumInterface _mediumInterface;

    };

    class Light : public utils::TaggedPointer<DiffuseAreaLight, PointLight, SpotLight, EnvironmentLight> {
    public:
        using TaggedPointer::TaggedPointer;

        RENDER_CPU_GPU
        Spectrum sampleLi(const Interaction &eye, Vector3F *wi, Float *pdf, Vector2F uv, Interaction *target);

        RENDER_CPU_GPU
        Float pdfLi(const Interaction &eye, const Vector3F &direction);

        RENDER_CPU_GPU
        Spectrum Le(const Ray &ray);

        RENDER_CPU_GPU
        LightSourceType getType() const;

        RENDER_CPU_GPU
        bool isDeltaType() const;

        // TODO delete
//        MediumInterface _mediumInterface;
    };

}

#endif //TUNAN_LIGHTS_H
