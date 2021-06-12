//
// Created by StormPhoenix on 2021/6/2.
//

#include <tunan/scene/cameras.h>

namespace RENDER_NAMESPACE {
    using namespace math;

    RENDER_CPU_GPU
    Camera::Camera(Transform cameraToWorld, Float hFov,
                   int resolutionWidth, int resolutionHeight,
                   Float nearClip, Float farClip)
            : _nearClip(nearClip), _farClip(farClip),
              _resolutionWidth(resolutionWidth), _resolutionHeight(resolutionHeight) {
        Float aspect = Float(resolutionWidth) / Float(resolutionHeight);
        _cameraToWorld = cameraToWorld;
        _worldToCamera = _cameraToWorld.inverse();

        _lensRadius = 0.000025;
        _rasterToCamera = (Transform::scale(resolutionWidth, resolutionHeight, 1.) *
                           Transform::scale(-0.5, 0.5 * aspect, 1.) *
                           Transform::translate(-1, 1.0 / aspect, 0) *
                           Transform::perspective(hFov, nearClip, farClip)).inverse();

        _cameraToRaster = _rasterToCamera.inverse();
        _origin = _cameraToWorld.transformPoint(Point3F(0., 0., 0.));

        Point3F filmMin = _rasterToCamera.transformPoint(Point3F(0, 0, 0));
        Point3F filmMax = _rasterToCamera.transformPoint(
                Point3F(resolutionWidth, resolutionHeight, 0));
        filmMin /= filmMin.z;
        filmMax /= filmMax.z;
        _filmPlaneArea = std::abs((filmMax.x - filmMin.x) * (filmMax.y - filmMin.y));
        _lensArea = Pi * _lensRadius * _lensRadius;
        _front = NORMALIZE(_cameraToWorld.transformPoint(Vector3F(0, 0, 1))
                           - _cameraToWorld.transformPoint(Vector3F(0, 0, 0)));
    }

    RENDER_CPU_GPU
    Ray Camera::generateRay(Float pixelX, Float pixelY) const {
        Float x = pixelX;
        Float y = pixelY;

        // Camera space
        Point3F pOrigin = Point3F(0, 0, 0);

        Point3F pTarget = _rasterToCamera.transformPoint(Point3F(x, y, 0));
        Vector3F rayDir = NORMALIZE(pTarget - pOrigin);

        // Convert to world space
        Point3F pOriginWorld = _cameraToWorld.transformPoint(pOrigin);
        Vector3F pDirWorld = NORMALIZE(_cameraToWorld.transformVector(rayDir));

        return Ray(pOriginWorld, pDirWorld);
    }

    RENDER_CPU_GPU
    Ray Camera::generateRayDifferential(Float pixelX, Float pixelY, const CameraSamples &cameraSamples) const {
        Float x = pixelX + cameraSamples.pixelJitter[0] - 0.5f;
        Float y = pixelY + cameraSamples.pixelJitter[1] - 0.5f;

        // Sample camera ray in camera space
        Point3F pOrigin = Point3F(0, 0, 0);
        if (_lensRadius > 0) {
            Vector2F diskSample = sampler::diskUniformSampling(cameraSamples.uvLens, _lensRadius);
            pOrigin = Point3F(diskSample.x, diskSample.y, 0);
        }
        Point3F pTarget = _rasterToCamera.transformPoint(Point3F(x, y, 0));
        Vector3F rayDir = NORMALIZE(pTarget - pOrigin);

        // Convert camera ray to world space
        Point3F pOriginWorld = _cameraToWorld.transformPoint(pOrigin);
        Vector3F pDirWorld = NORMALIZE(_cameraToWorld.transformVector(rayDir));

        return Ray(pOriginWorld, pDirWorld);
    }

    RENDER_CPU_GPU
    void Camera::pdfWe(const Ray &ray, Float &pdfPos, Float &pdfDir) const {
        Vector3F rayDir = NORMALIZE(ray.getDirection());
        Float cosine = DOT(rayDir, _front);
        if (cosine <= 0) {
            pdfPos = 0.;
            pdfDir = 0.;
            return;
        }

        // Compute raster position
        Point3F pFocus = (1 / cosine) * rayDir + ray.getOrigin();
        Point3F pRaster = _cameraToRaster.transformPoint(_worldToCamera.transformPoint(pFocus));
        // Check range
        if (pRaster.x < 0 || pRaster.x >= _resolutionWidth ||
            pRaster.y < 0 || pRaster.y >= _resolutionHeight) {
            pdfPos = 0.0;
            pdfDir = 0.0;
            return;
        }

        // 计算 pdfPos pdfDir
        pdfPos = 1. / (_lensArea);
        pdfDir = 1. / (_filmPlaneArea * cosine * cosine * cosine);
        return;
    }

    RENDER_CPU_GPU
    Spectrum Camera::We(const Ray &ray, Point2F *const filmPosition) const {
        Vector3F rayDir = NORMALIZE(ray.getDirection());
        Float cosine = DOT(rayDir, _front);
        if (cosine <= 0) {
            return Spectrum(0.);
        }

        // Compute raster position
        Point3F pFocus = (1 / cosine) * rayDir + ray.getOrigin();
        Point3F pRaster = _cameraToRaster.transformPoint(_worldToCamera.transformPoint(pFocus));

        // Check range
        if (pRaster.x < 0 || pRaster.x >= _resolutionWidth ||
            pRaster.y < 0 || pRaster.y >= _resolutionHeight) {
            return Spectrum(0.);
        }

        if (filmPosition != nullptr) {
            (*filmPosition) = Point2F(pRaster.x, pRaster.y);
        }
        /*
         * 1) p(w) = dist^2 / (A * cosine)
         * 2) dist * cosine = _focal
         * 3) W_e = p(w) / (PI * lensRadius^2 * cosine)
         * 4) A is area of image plane
         * 5) W_e = _focal^2 / (cosine^4 * A * PI * lensRadius^2)

         * Float dist = _focal / cosine;
         * Float pW = dist * dist / (_area * cosine);
         * Float weight = pW / (PI * _lensRadius * _lensRadius * cosine);
        */
        Float weight = 1. / (_filmPlaneArea * _lensArea * std::pow(cosine, 4));
        return Spectrum(weight);
    }

    RENDER_CPU_GPU
    Point2I Camera::worldToRaster(const Point3F &point) const {
        Point3F cameraPoint = _worldToCamera.transformPoint(point);
        Point3F pRaster = _cameraToRaster.transformPoint(cameraPoint);
        return Point2I(pRaster.x, pRaster.y);
    }
}