//
// Created by Storm Phoenix on 2021/6/8.
//

#include <tunan/scene/lights.h>

namespace RENDER_NAMESPACE {

    DiffuseAreaLight::DiffuseAreaLight(const Spectrum &intensity, Shape shape,
                                       const MediumInterface *mediumInterface, bool twoSided) :
            _intensity(intensity), _shape(shape), _type(Area), _mediumInterface(mediumInterface), _twoSided(twoSided) {}

    RENDER_CPU_GPU
    Spectrum DiffuseAreaLight::sampleLi(const Interaction &eye, Vector3F *wi, Float *pdf, Vector2F uv) {
        assert(!_shape.nullable());
        // 从 eye 出发采样一条射线，返回与 shape 的交点
        SurfaceInteraction si = _shape.sample(eye.p, pdf, uv);
        (*wi) = NORMALIZE(si.p - eye.p);
        return L(si, -(*wi));
    }

    RENDER_CPU_GPU
    Spectrum DiffuseAreaLight::L(const Interaction &interaction, const Vector3F &wo) const {
        Float cosTheta = DOT(interaction.ng, wo);
        return (_twoSided || cosTheta > 0) ? _intensity : Spectrum(0.0);
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

    RENDER_CPU_GPU
    Spectrum Light::sampleLi(const Interaction &eye, Vector3F *wi, Float *pdf, Vector2F uv) {
        auto func = [&](auto ptr) {
            return ptr->sampleLi(eye, wi, pdf, uv);
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