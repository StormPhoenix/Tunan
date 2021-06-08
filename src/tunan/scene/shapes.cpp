//
// Created by Storm Phoenix on 2021/6/8.
//

#include <tunan/math.h>
#include <tunan/scene/shapes.h>
#include <tunan/sampler/samplers.h>

namespace RENDER_NAMESPACE {
    RENDER_CPU_GPU
    Triangle::Triangle(const Vector3F &p0, const Vector3F &p1, const Vector3F &p2,
                       const Normal3F &n0, const Normal3F &n1, const Normal3F &n2)
            : _p0(p0), _p1(p1), _p2(p2), _n0(n0), _n1(n1), _n2(n2) {
        nsValid = !(LENGTH(_n0) == 0 || LENGTH(_n1) == 0 || LENGTH(_n2) == 0);
    }

    RENDER_CPU_GPU
    bool Triangle::intersect(Ray &ray, SurfaceInteraction &si,
                             Float minStep, Float maxStep) const {
        /* transform the ray direction, let it to point to z-axis */

        // move ray's origin to (0, 0, 0)
        Vector3F p0 = _p0 - ray.getOrigin();
        Vector3F p1 = _p1 - ray.getOrigin();
        Vector3F p2 = _p2 - ray.getOrigin();

        // swap axis
        int zAxis = math::maxAbsComponentIndex(ray.getDirection());
        int xAxis = zAxis + 1 == 3 ? 0 : zAxis + 1;
        int yAxis = xAxis + 1 == 3 ? 0 : xAxis + 1;

        Vector3F dir = math::swapComponent(ray.getDirection(), xAxis, yAxis, zAxis);
        p0 = math::swapComponent(p0, xAxis, yAxis, zAxis);
        p1 = math::swapComponent(p1, xAxis, yAxis, zAxis);
        p2 = math::swapComponent(p2, xAxis, yAxis, zAxis);

        // shear direction to z-axis
        Float shearX = -dir[0] / dir[2];
        Float shearY = -dir[1] / dir[2];

        // transform points to z-axis
        p0[0] += shearX * p0[2];
        p0[1] += shearY * p0[2];
        p1[0] += shearX * p1[2];
        p1[1] += shearY * p1[2];
        p2[0] += shearX * p2[2];
        p2[1] += shearY * p2[2];

        // calculate barycentric
        Float e0 = p1.y * p0.x - p1.x * p0.y;
        Float e1 = p2.y * p1.x - p2.x * p1.y;
        Float e2 = p0.y * p2.x - p0.x * p2.y;

        // check double precision if barycentric is zero
        // Fall back to double precision test at triangle edges
        if (sizeof(Float) == sizeof(float) &&
            (e0 == 0.0f || e1 == 0.0f || e2 == 0.0f)) {
            double p1yp0x = (double) p1.y * (double) p0.x;
            double p1xp0y = (double) p1.x * (double) p0.y;
            e0 = (float) (p1yp0x - p1xp0y);
            double p2yp1x = (double) p2.y * (double) p1.x;
            double p2xp1y = (double) p2.x * (double) p1.y;
            e1 = (float) (p2yp1x - p2xp1y);
            double p0yp2x = (double) p0.y * (double) p2.x;
            double p0xp2y = (double) p0.x * (double) p2.y;
            e2 = (float) (p0yp2x - p0xp2y);
        }

        if ((e0 > 0 || e1 > 0 || e2 > 0) && (e0 < 0 || e1 < 0 || e2 < 0)) {
            // intersection point doesn't fall in triangle area.
            return false;
        }

        Float sum = e0 + e1 + e2;

        if (sum == 0) {
            return false;
        }

        // interpolate step * sum
        Float shearZ = 1 / dir[2];
        p0[2] *= shearZ;
        p1[2] *= shearZ;
        p2[2] *= shearZ;
        Float sumMulStep = e0 * p2.z + e1 * p0.z + e2 * p1.z;

        // make sure step > 0 and step < t_max
        if (sum > 0 && (sumMulStep <= sum * minStep || sumMulStep >= sum * maxStep)) {
            return false;
        } else if (sum < 0 && (sumMulStep >= sum * minStep || sumMulStep <= sum * maxStep)) {
            return false;
        }

        Float invSum = 1. / sum;
        Float b0 = e0 * invSum;
        Float b1 = e1 * invSum;
        Float b2 = e2 * invSum;
        Float step = sumMulStep * invSum;

        Float maxZ = math::maxAbsComponent(Vector3F(p0.z, p1.z, p2.z));
        Float deltaZ = math::gamma(3) * maxZ;

        Float maxX = math::maxAbsComponent(Vector3F(p0.x, p1.x, p2.x));
        Float maxY = math::maxAbsComponent(Vector3F(p0.y, p1.y, p2.y));

        Float deltaX = math::gamma(5) * (maxX + maxZ);
        Float deltaY = math::gamma(5) * (maxY + maxZ);

        Float deltaE = 2 * (math::gamma(2) * maxX * maxY + deltaY * maxX + deltaX * maxY);

        Float maxE = math::maxAbsComponent(Vector3F(e0, e1, e2));
        Float deltaT = 3 * (math::gamma(3) * maxE * maxZ + deltaE * maxZ + deltaZ * maxE) * std::abs(invSum);
        if (step <= deltaT) {
            return false;
        }

        // calculate float-error
        Float zError = math::gamma(7) * (std::abs(b0 * p2.z) + std::abs(b1 * p0.z) + std::abs(b2 * p1.z));
        Float xError = math::gamma(7) * (std::abs(b0 * p2.x) + std::abs(b1 * p0.x) + std::abs(b2 * p1.x));
        Float yError = math::gamma(7) * (std::abs(b0 * p2.y) + std::abs(b1 * p0.y) + std::abs(b2 * p1.y));
        Vector3F error = Vector3F(xError, yError, zError);

        si.p = b1 * _p0 + b2 * _p1 + b0 * _p2;

        Vector3F ng, ns;
        ng = NORMALIZE(CROSS(_p1 - _p0, _p2 - _p0));
        if (!nsValid) {
            ns = ng;
        } else {
            ns = b1 * _n0 + b2 * _n1 + b0 * _n2;
            if (DOT(ns, ng) < 0) {
                ng *= -1;
            }
            // TODO delete
//            if (Config::Tracer::strictNormals) {
//                ns = ng;
//            }
        }
        si.ng = ng;
        si.ns = ns;

        si.wo = -NORMALIZE(ray.getDirection());
//        si.u = _uv1.x * b1 + _uv2.x * b2 + _uv3.x * b0;
//        si.v = _uv1.y * b1 + _uv2.y * b2 + _uv3.y * b0;
        si.error = error;
        ray.setStep(step);
        return true;
    }

    RENDER_CPU_GPU
    Float Triangle::area() const {
        return 0.5 * LENGTH(CROSS(_p1 - _p0, _p2 - _p0));
    }

    RENDER_CPU_GPU
    SurfaceInteraction Triangle::sampleSurfacePoint(Float *pdf, Vector2F uv) const {
        // Uniform sampling triangle
        Vector2F barycentric = sampler::triangleUniformSampling(uv);

        SurfaceInteraction si;
        Vector3F p = _p0 * barycentric[0] + _p1 * barycentric[1] +
                     _p2 * (1 - barycentric[0] - barycentric[1]);
        si.p = p;

        // error bound
        Point3F pAbsSum = ABS(barycentric[0] * _p0) + ABS(barycentric[1] * _p1) +
                          ABS((1 - barycentric[0] - barycentric[1]) * _p2);
        si.error = math::gamma(6) * Vector3F(pAbsSum.x, pAbsSum.y, pAbsSum.z);

        // geometry normal
        Vector3F ng = NORMALIZE(CROSS(_p1 - _p0, _p2 - _p0));

        // shading normal
        Vector3F ns(0);
        if (!nsValid) {
            ns = ng;
        } else {
            ns = _n0 * barycentric[0] + _n1 * barycentric[1] +
                 _n2 * (1 - barycentric[0] - barycentric[1]);

            if (DOT(ns, ng) < 0) {
                // correct geometry normal
                ng *= -1;
            }
        }

        if (pdf != nullptr) {
            (*pdf) = 1. / area();
        }

        si.ng = ng;
        si.ns = ns;
        return si;
    }

    RENDER_CPU_GPU
    SurfaceInteraction Triangle::sample(const Point3F &eye, Float *pdf, Vector2F uv) const {
        Float density = 0.;
        SurfaceInteraction si = sampleSurfacePoint(&density, uv);
        Vector3F wi = si.p - eye;
        Float distance = LENGTH(wi);
        if (distance == 0.) {
            (*pdf) = 0.;
        } else {
            wi = NORMALIZE(wi);
            Float cosTheta = ABS_DOT(-wi, si.ng);
            (*pdf) = density * (distance * distance) / cosTheta;
            if (std::isinf(*pdf)) {
                (*pdf) = 0.;
            }
        }
        return si;
    }

    RENDER_CPU_GPU
    Float Triangle::surfaceInteractionPdf(const Point3F &eye, const Vector3F &direction) const {
        // build a ray to test interaction
        Ray ray(eye, direction);
        SurfaceInteraction si;
        bool foundIntersection = intersect(ray, si, ray.getMinStep(), ray.getStep());
        if (foundIntersection) {
            // convert density to pdf
            Float distance = LENGTH(si.p - eye);
            Float cosine = ABS_DOT(direction, si.ng);
            if (std::abs(cosine - 0) < Epsilon) {
                return 0;
            }
            return pow(distance, 2) / (cosine * area());
        } else {
            return 0;
        }
    }

    RENDER_CPU_GPU
    bool Shape::intersect(Ray &ray, SurfaceInteraction &si, Float minStep, Float maxStep) const {
        auto func = [&](auto ptr) {
            return ptr->intersect(ray, si, minStep, maxStep);
        };
        return proxyCall(func);
    }

    RENDER_CPU_GPU
    SurfaceInteraction Shape::sample(const Point3F &eye, Float *pdf, Vector2F uv) const {
        auto func = [&](auto ptr) {
            return ptr->sample(eye, pdf, uv);
        };
        return proxyCall(func);
    }

    RENDER_CPU_GPU
    Float Shape::surfaceInteractionPdf(const Point3F &eye, const Vector3F &direction) const {
        auto func = [&](auto ptr) {
            return ptr->surfaceInteractionPdf(eye, direction);
        };
        return proxyCall(func);
    }

    RENDER_CPU_GPU
    SurfaceInteraction Shape::sampleSurfacePoint(Float *pdf, Vector2F uv) const {
        auto func = [&](auto ptr) {
            return ptr->sampleSurfacePoint(pdf, uv);
        };
        return proxyCall(func);
    }
}