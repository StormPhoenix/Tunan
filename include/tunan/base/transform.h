//
// Created by StormPhoenix on 2021/6/2.
//

#ifndef TUNAN_TRANSFORM_H
#define TUNAN_TRANSFORM_H

#include <tunan/common.h>
#include <tunan/base/math.h>

namespace RENDER_NAMESPACE {
    namespace base {
        class Transform {
        public:
            RENDER_CPU_GPU
            Transform();

            RENDER_CPU_GPU
            Transform(Matrix4f transformMatrix);

            RENDER_CPU_GPU
            Point3f transformPoint(const Point3f &p) const;

            RENDER_CPU_GPU
            Vector3f transformVector(const Vector3f &v) const;

            RENDER_CPU_GPU
            Normal3f transformNormal(const Normal3f &n) const;

            RENDER_CPU_GPU
            Matrix4f mat() const;

            RENDER_CPU_GPU
            Transform inverse() const;

            RENDER_CPU_GPU
            Transform operator*(const Transform &t) const;

            RENDER_CPU_GPU
            static Transform perspective(Float fov, Float nearClip, Float farClip);

            RENDER_CPU_GPU
            static Transform translate(Float x, Float y, Float z);

            RENDER_CPU_GPU
            static Transform scale(Float sx, Float sy, Float sz);

            RENDER_CPU_GPU
            static Transform lookAt(Point3f origin, Point3f target, Vector3f up);

        private:
            bool _identity = false;
            Matrix4f _transformMatrix;
        };
    }
}

#endif //TUNAN_TRANSFORM_H
