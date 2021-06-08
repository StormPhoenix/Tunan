//
// Created by Storm Phoenix on 2021/6/8.
//

#ifndef TUNAN_SHAPES_H
#define TUNAN_SHAPES_H

#include <tunan/common.h>
#include <tunan/base/interactions.h>
#include <tunan/utils/TaggedPointer.h>

namespace RENDER_NAMESPACE {
    using utils::TaggedPointer;
    using base::SurfaceInteraction;

    class Triangle {
    public:
        RENDER_CPU_GPU
        Triangle(const Vector3F &p0, const Vector3F &p1, const Vector3F &p2,
                 const Normal3F &n0, const Normal3F &n1, const Normal3F &n2);

        RENDER_CPU_GPU
        bool intersect(Ray &ray, SurfaceInteraction &si, Float minStep, Float maxStep) const;

        RENDER_CPU_GPU
        SurfaceInteraction sample(const Point3F &eye, Float *pdf, Vector2F uv) const;

        RENDER_CPU_GPU
        Float surfaceInteractionPdf(const Point3F &eye, const Vector3F &direction) const;

        RENDER_CPU_GPU
        SurfaceInteraction sampleSurfacePoint(Float *pdf, Vector2F uv) const;

    private:
        RENDER_CPU_GPU
        Float area() const;

    private:
        const Vector3F _p0, _p1, _p2;
        const Normal3F _n0, _n1, _n2;
        bool nsValid = true;
    };


    class Shape : public TaggedPointer<Triangle> {
    public:
        using TaggedPointer::TaggedPointer;

        RENDER_CPU_GPU
        bool intersect(Ray &ray, SurfaceInteraction &si, Float minStep, Float maxStep) const;

        RENDER_CPU_GPU
        SurfaceInteraction sample(const Point3F &eye, Float *pdf, Vector2F uv) const;

        RENDER_CPU_GPU
        Float surfaceInteractionPdf(const Point3F &eye, const Vector3F &direction) const;

        RENDER_CPU_GPU
        SurfaceInteraction sampleSurfacePoint(Float *pdf, Vector2F uv) const;
    };
}

#endif //TUNAN_SHAPES_H
