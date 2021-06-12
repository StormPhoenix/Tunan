//
// Created by StormPhoenix on 2021/6/2.
//

#ifndef TUNAN_CAMERAS_H
#define TUNAN_CAMERAS_H

#include <tunan/common.h>
#include <tunan/scene/Ray.h>
#include <tunan/sampler/samplers.h>
#include <tunan/base/spectrum.h>
#include <tunan/base/transform.h>

namespace RENDER_NAMESPACE {
    using base::Spectrum;
    using base::Transform;
    using sampler::Sampler;

    typedef struct CameraSamples {
        Point2F uvLens;
        Point2F pixelJitter;
    } CameraSamples;

    class Camera {
    public:
        RENDER_CPU_GPU
        Camera(Transform cameraToWorld, Float hFov,
               int resolutionWidth, int resolutionHeight,
               Float nearClip = 1.0, Float farClip = 1000);

        RENDER_CPU_GPU
        Ray generateRay(Float pixelX, Float pixelY) const;

        RENDER_CPU_GPU
        Ray generateRayDifferential(Float pixelX, Float pixelY, const CameraSamples &cameraSamples) const;

        RENDER_CPU_GPU
        void pdfWe(const Ray &ray, Float &pdfPos, Float &pdfDir) const;

        RENDER_CPU_GPU
        Point2I worldToRaster(const Point3F &point) const;

        RENDER_CPU_GPU
        const Vector3F &getFront() const {
            return _front;
        }

        RENDER_CPU_GPU
        int getWidth() const {
            return _resolutionWidth;
        }

        RENDER_CPU_GPU
        int getHeight() const {
            return _resolutionHeight;
        }

    private:
        /**
         * Ray importance and film position computation
         * @param ray
         * @param filmPosition
         * @return
         */
        RENDER_CPU_GPU
        Spectrum We(const Ray &ray, Point2F *const filmPosition) const;

    public:
        Vector3F _origin;
        Vector3F _front;
        Float _filmPlaneArea;
        Float _lensArea;
        Float _lensRadius = 0.00025;

        int _resolutionWidth;
        int _resolutionHeight;

        Float _nearClip;
        Float _farClip;

        Transform _cameraToWorld;
        Transform _worldToCamera;
        Transform _rasterToCamera;
        Transform _cameraToRaster;
    };

}

#endif //TUNAN_CAMERAS_H
