//
// Created by StormPhoenix on 2021/6/2.
//

#include <tunan/base/transform.h>

namespace RENDER_NAMESPACE {
    namespace base {
        RENDER_CPU_GPU
        Transform::Transform() : _identity(true) {}

        RENDER_CPU_GPU
        Transform::Transform(Matrix4F transformMatrix) :
                _transformMatrix(transformMatrix),
                _identity(false) {}

        RENDER_CPU_GPU
        Vector3F Transform::transformPoint(const Point3F &p) const {
            if (_identity) {
                return p;
            }

            Point4F ret = _transformMatrix * Vector4d(p, 1.);
            if (ret[3] == 1) {
                return Vector3F(ret[0], ret[1], ret[2]);
            } else {
                return Vector3F(ret[0], ret[1], ret[2]) / ret[3];
            }
        }

        RENDER_CPU_GPU
        Vector3F Transform::transformVector(const Vector3F &v) const {
            if (_identity) {
                return v;
            }
            return _transformMatrix * Vector4d(v, 0.);
        }

        RENDER_CPU_GPU
        Vector3F Transform::transformNormal(const Normal3F &n) const {
            Normal3F normal = NORMALIZE(n);
            if (_identity) {
                return normal;
            }
            Vector3F ret = INVERSE_TRANSPOSE(_transformMatrix) * Vector4d(normal, 0.0f);
            return NORMALIZE(ret);
        }

        RENDER_CPU_GPU
        Matrix4F Transform::mat() const {
            return _transformMatrix;
        }

        RENDER_CPU_GPU
        Transform Transform::inverse() const {
            if (_identity) {
                return Transform();
            }

            return INVERSE(_transformMatrix);
        }

        RENDER_CPU_GPU
        Transform Transform::operator*(const Transform &t) const {
            return Transform(_transformMatrix * t._transformMatrix);
        }

        RENDER_CPU_GPU
        Transform Transform::perspective(Float fov, Float nearClip, Float farClip) {
            Float invTan = 1.0 / std::tan(RADIANS(fov * 0.5));
            Matrix4F perMat;

            perMat[0][0] = invTan;
            perMat[1][0] = 0;
            perMat[2][0] = 0;
            perMat[3][0] = 0;

            perMat[0][1] = 0;
            perMat[1][1] = invTan;
            perMat[2][1] = 0;
            perMat[3][1] = 0;

            perMat[0][2] = 0;
            perMat[1][2] = 0;
            perMat[2][2] = farClip / (farClip - nearClip);
            perMat[3][2] = -farClip * nearClip / (farClip - nearClip);

            perMat[0][3] = 0;
            perMat[1][3] = 0;
            perMat[2][3] = 1;
            perMat[3][3] = 0;

            return Transform(perMat);
        }

        RENDER_CPU_GPU
        Transform Transform::translate(Float x, Float y, Float z) {
            Vector3F offset(x, y, z);
            Matrix4F m(1.0);
            m = TRANSLATE(m, offset);
            return Transform(m);
        }

        RENDER_CPU_GPU
        Transform Transform::scale(Float sx, Float sy, Float sz) {
            Vector3F factor(sx, sy, sz);
            Matrix4F m(1.0);
            m = SCALE(m, factor);
            return Transform(m);
        }

        RENDER_CPU_GPU
        Transform Transform::lookAt(Point3F origin, Point3F target, Vector3F up) {
            Matrix4F mat;
            mat[3][0] = origin[0];
            mat[3][1] = origin[1];
            mat[3][2] = origin[2];
            mat[3][3] = 1;

            Vector3F forward = NORMALIZE(target - origin);
            Vector3F left = CROSS(up, forward);
            Vector3F realUp = CROSS(forward, left);
            mat[0][0] = left[0];
            mat[0][1] = left[1];
            mat[0][2] = left[2];
            mat[0][3] = 0;

            mat[1][0] = realUp[0];
            mat[1][1] = realUp[1];
            mat[1][2] = realUp[2];
            mat[1][3] = 0;

            mat[2][0] = forward[0];
            mat[2][1] = forward[1];
            mat[2][2] = forward[2];
            mat[2][3] = 0;

            return Transform(mat);
        }
    }
}