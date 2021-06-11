//
// Created by Storm Phoenix on 2021/6/8.
//

#include <tunan/scene/lights.h>

namespace RENDER_NAMESPACE {

    DiffuseAreaLight::DiffuseAreaLight(const Spectrum &radiance, Shape shape,
                                       MediumInterface mediumInterface, bool twoSided) :
            _radiance(radiance), _shape(shape), _type(Area), _mediumInterface(mediumInterface), _twoSided(twoSided) {}

    RENDER_CPU_GPU
    Spectrum DiffuseAreaLight::sampleLi(const Interaction &eye, Vector3F *wi, Float *pdf,
                                        Vector2F uv, Interaction *target) {
        assert(!_shape.nullable());
        // 从 eye 出发采样一条射线，返回与 shape 的交点
        SurfaceInteraction si = _shape.sample(eye.p, pdf, uv);
        (*wi) = NORMALIZE(si.p - eye.p);
        if (target != nullptr) {
            target->p = si.p;
            target->ng = si.ng;
            target->wo = -(*wi);
            target->error = si.error;
            target->mediumInterface = _mediumInterface;
        }
        return L(si, -(*wi));
    }

    RENDER_CPU_GPU
    Spectrum DiffuseAreaLight::L(const Interaction &interaction, const Vector3F &wo) const {
        Float cosTheta = DOT(interaction.ng, wo);
        return (_twoSided || cosTheta > 0) ? _radiance : Spectrum(0.0);
    }

    RENDER_CPU_GPU
    Float DiffuseAreaLight::pdfLi(const Interaction &eye, const Vector3F &dir) {
        assert(_shape != nullptr);
        return _shape.surfaceInteractionPdf(eye.p, dir);
    }

    RENDER_CPU_GPU
    LightSourceType DiffuseAreaLight::getType() const {
        return _type;
    }

    PointLight::PointLight(const Spectrum &intensity, Transform lightToWorld, MediumInterface mediumInterface) :
            _type(Delta_Position), _mediumInterface(mediumInterface), _intensity(intensity),
            _lightToWorld(lightToWorld) {
        _center = lightToWorld.transformPoint(Point3F(0));
    }

    RENDER_CPU_GPU
    Spectrum PointLight::sampleLi(const Interaction &eye, Vector3F *wi, Float *pdf, Vector2F uv, Interaction *target) {
        (*wi) = NORMALIZE(_center - eye.p);
        (*pdf) = 1;

        Vector3F samplePoint = _center;
        Vector3F sampleDir = NORMALIZE(_center - eye.p);
        Normal3F sampleNormal = -sampleDir;
        if (target != nullptr) {
            target->p = samplePoint;
            target->ng = sampleNormal;
            target->wo = -(*wi);
            target->error = Vector3F(0);
            target->mediumInterface = _mediumInterface;
        }

        Float distance = LENGTH(_center - eye.p);
        return _intensity / (distance * distance);
    }

    RENDER_CPU_GPU
    Float PointLight::pdfLi(const Interaction &eye, const Vector3F &dir) {
        return 0.0;
    }

    RENDER_CPU_GPU
    Spectrum Light::sampleLi(const Interaction &eye, Vector3F *wi, Float *pdf, Vector2F uv, Interaction *target) {
        auto func = [&](auto ptr) {
            return ptr->sampleLi(eye, wi, pdf, uv, target);
        };
        return proxyCall(func);
    }

    RENDER_CPU_GPU
    Float Light::pdfLi(const Interaction &eye, const Vector3F &direction) {
        auto func = [&](auto ptr) {
            return ptr->pdfLi(eye, direction);
        };
        return proxyCall(func);
    }

    RENDER_CPU_GPU
    LightSourceType Light::getType() const {
        auto func = [&](auto ptr) {
            return ptr->getType();
        };
        return proxyCall(func);
    }

    RENDER_CPU_GPU
    bool Light::isDeltaType() const {
        auto func = [&](auto ptr) {
            LightSourceType type = ptr->getType();
            return (type & (Delta_Direction | Delta_Position)) > 0;
        };
        return proxyCall(func);
    }
}