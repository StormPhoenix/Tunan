//
// Created by StormPhoenix on 2021/6/2.
//

#ifndef TUNAN_TRANSFORM_H
#define TUNAN_TRANSFORM_H

#include <tunan/common.h>
#include <tunan/math.h>

namespace RENDER_NAMESPACE {
    namespace base {
        class Transform {
        public:
            RENDER_CPU_GPU
            Transform();

            RENDER_CPU_GPU
            Transform(Matrix4F transformMatrix);

            RENDER_CPU_GPU
            Vector3F transformPoint(const Point3F &p) const;

            RENDER_CPU_GPU
            Vector3F transformVector(const Vector3F &v) const;

            RENDER_CPU_GPU
            Vector3F transformNormal(const Vector3F &n) const;

            RENDER_CPU_GPU
            Matrix4F mat() const;

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
            static Transform lookAt(Point3F origin, Point3F target, Vector3F up);

        private:
            bool _identity = false;
            Matrix4F _transformMatrix;
        };
    }
}

#endif //TUNAN_TRANSFORM_H
