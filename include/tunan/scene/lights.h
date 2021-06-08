//
// Created by Storm Phoenix on 2021/6/8.
//

#ifndef TUNAN_LIGHTS_H
#define TUNAN_LIGHTS_H

#include <tunan/common.h>
#include <tunan/scene/shapes.h>
#include <tunan/base/mediums.h>
#include <tunan/base/spectrum.h>
#include <tunan/base/interactions.h>
#include <tunan/utils/TaggedPointer.h>

namespace RENDER_NAMESPACE {
    using base::MediumInterface;
    using base::Interaction;
    using base::Spectrum;

    typedef enum LightSourceType {
        Delta_Position = 1 << 0,
        Delta_Direction = 1 << 1,
        Area = 1 << 2,
        Environment = 1 << 3
    } LightSourceType;

    class DiffuseAreaLight {
    public:
        DiffuseAreaLight(const Spectrum &intensity, Shape shape,
                         const MediumInterface *mediumBoundary, bool twoSided = false);

        RENDER_CPU_GPU
        Spectrum sampleLi(const Interaction &eye, Vector3F *wi, Float *pdf, Vector2F uv);

        RENDER_CPU_GPU
        Spectrum L(const Interaction &interaction, const Vector3F &wo) const;

        RENDER_CPU_GPU
        Float pdfLi(const Interaction &eye, const Vector3F &dir);

        RENDER_CPU_GPU
        LightSourceType getType() const;

    protected:
        Shape _shape;
        bool _twoSided = false;
        Spectrum _intensity;
        LightSourceType _type = Area;
        const MediumInterface *_mediumInterface = nullptr;
    };

    class Light : public utils::TaggedPointer<DiffuseAreaLight> {
    public:
        using TaggedPointer::TaggedPointer;

        RENDER_CPU_GPU
        Spectrum sampleLi(const Interaction &eye, Vector3F *wi, Float *pdf, Vector2F uv);

        RENDER_CPU_GPU
        Float pdfLi(const Interaction &eye, const Vector3F &direction);

        RENDER_CPU_GPU
        LightSourceType getType() const;

        RENDER_CPU_GPU
        bool isDeltaType() const;

        // TODO delete
//        MediumInterface _mediumInterface;
    };

}

#endif //TUNAN_LIGHTS_H
