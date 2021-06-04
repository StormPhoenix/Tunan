//
// Created by StormPhoenix on 2021/5/31.
//

#ifndef TUNAN_MATH_H
#define TUNAN_MATH_H

#include <cassert>
#include <tunan/common.h>
#include <cmath>

/*
using Vector2d = glm::dvec2;
using Vector2f = glm::vec2;
using Vector2i = glm::int2;

using Vector3i = glm::int3;
using Vector3d = glm::dvec3;
using Vector3f = glm::vec3;

using Vector4d = glm::dvec4;
using Vector4i = glm::int4;
using Vector4f = glm::vec4;

using Color = glm::int3;

using Matrix4d = glm::dmat4x4;
using Matrix3d = glm::dmat3x3;

using Matrix4f = glm::mat4x4;
using Matrix3f = glm::mat3x3;
 */

/*
#if defined(_RENDER_DATA_DOUBLE_)
using Vector4F = Vector4d;
using Vector3F = Vector3d;
using Vector2F = Vector2d;
using Matrix4F = Matrix4d;
using Matrix3F = Matrix3d;
#else
using Vector4F = Vector4f;
using Vector3F = Vector3f;
using Vector2F = Vector2f;
using Matrix4F = Matrix4f;
using Matrix3F = Matrix3f;
#endif
 */

#define RADIANS(radius) glm::radians(radius)
#define ROTATE(matrix, radius, axis) glm::rotate(matrix, glm::radians(radius), axis)
#define TRANSLATE(matrix, offset) glm::translate(matrix, offset)
#define SCALE(matrix, factor) glm::scale(matrix, factor)
#define INVERSE(matrix) glm::inverse(matrix)
#define INVERSE_TRANSPOSE(matrix) glm::inverseTranspose(matrix)
#define DETERMINANT(x) glm::determinant(x)
#define PERSPECTIVE(fovy, aspect, zNear, zFar) glm::perspective(glm::radians(fovy), aspect, zNear, zFar)
#define DOT(x, y) glm::dot(x, y)
#define ABS(x) glm::abs(x)
#define ABS_DOT(x, y) std::abs(glm::dot(x, y))
#define NORMALIZE(x) glm::normalize(x)
#define LENGTH(x) glm::length(x)
#define CROSS(x, y) glm::cross(x, y)
#define LOOK_AT(eye, center, up) glm::lookAt(eye, center, up)

namespace RENDER_NAMESPACE {
    namespace base {
#ifdef __RENDER_GPU_MODE__

#define Infinity std::numeric_limits<Float>::infinity()
#define Epsilon std::numeric_limits<Float>::epsilon() * 0.5f
#define MaxFloat std::numeric_limits<Float>::max()

#define doubleOneMinusEpsilon 0x1.fffffffffffffp-1
#define floatOneMinusEpsilon float(0x1.fffffep-1)

#if defined(_RENDER_DATA_DOUBLE_)
#define OneMinusEpsilon doubleOneMinusEpsilon
#else
#define OneMinusEpsilon floatOneMinusEpsilon
#endif // _RENDER_DATA_DOUBLE_

#else

        static constexpr Float Infinity = std::numeric_limits<Float>::infinity();
        static constexpr Float Epsilon = std::numeric_limits<Float>::epsilon() * 0.5;
        static constexpr Float MaxFloat = std::numeric_limits<Float>::max();

        static constexpr double doubleOneMinusEpsilon = 0x1.fffffffffffffp-1;
        static constexpr float floatOneMinusEpsilon = 0x1.fffffep-1;

#if defined(_RENDER_DATA_DOUBLE_)
        static constexpr double OneMinusEpsilon = doubleOneMinusEpsilon;
#else
        static constexpr float OneMinusEpsilon = floatOneMinusEpsilon;
#endif // _RENDER_DATA_DOUBLE_

#endif // __RENDER_GPU_MODE__

#ifdef __RENDER_GPU_MODE__
#define Pi Float(3.14159265358979323846)
#define Inv_Pi Float(0.31830988618379067154)
#define Inv_2Pi Float(0.15915494309189533577)
#define Inv_4Pi Float(0.07957747154594766788)
#else
        constexpr Float Pi = 3.14159265358979323846;
        constexpr Float Inv_Pi = 0.31830988618379067154;
        constexpr Float Inv_2Pi = 0.15915494309189533577;
        constexpr Float Inv_4Pi = 0.07957747154594766788;
#endif // __RENDER_GPU_MODE__

        template<typename T>
        class Vector4;

        template<typename T>
        class Vector3 {
        public:
            RENDER_CPU_GPU
            Vector3() {
                x = y = z = 0;
            }

            RENDER_CPU_GPU
            Vector3(T v) {
                x = y = z = v;
            }

            RENDER_CPU_GPU
            Vector3(T xx, T yy, T zz) {
                x = xx;
                y = yy;
                z = zz;
            }

            RENDER_CPU_GPU
            Vector3(const Vector4<T> &p) {
                x = p.x;
                y = p.y;
                z = p.z;
            }

            RENDER_CPU_GPU
            T operator[](int i) const {
                CHECK(i >= 0 && i < 3);
                if (i == 0) {
                    return x;
                } else if (i == 1) {
                    return y;
                } else {
                    return z;
                }
            }

            RENDER_CPU_GPU
            T &operator[](int i) {
                CHECK(i >= 0 && i < 3);
                if (i == 0) {
                    return x;
                } else if (i == 1) {
                    return y;
                } else {
                    return z;
                }
            }

            T x, y, z;
        };

        template<typename T>
        class Vector4 {
        public:
            RENDER_CPU_GPU
            Vector4() {
                x = y = z = w = 0;
            }

            RENDER_CPU_GPU
            Vector4(T v) {
                x = y = z = w = v;
            }

            RENDER_CPU_GPU
            Vector4(T xx, T yy, T zz, T ww) {
                x = xx;
                y = yy;
                z = zz;
                w = ww;
            }

            RENDER_CPU_GPU
            Vector4(const Vector3<T> &p, T v) {
                x = p.x;
                y = p.y;
                z = p.z;
                w = v;
            }

            RENDER_CPU_GPU
            T operator[](int i) const {
                CHECK(i >= 0 && i < 4);
                if (i == 0) {
                    return x;
                } else if (i == 1) {
                    return y;
                } else if (i == 2) {
                    return z;
                } else {
                    return w;
                }
            }

            RENDER_CPU_GPU
            T &operator[](int i) {
                CHECK(i >= 0 && i < 4);
                if (i == 0) {
                    return x;
                } else if (i == 1) {
                    return y;
                } else if (i == 2) {
                    return z;
                } else {
                    return w;
                }
            }

            T x, y, z, w;
        };

        template<typename T>
        class Point3 {
        public:
            RENDER_CPU_GPU
            Point3() {
                x = y = z = 0;
            }

            RENDER_CPU_GPU
            Point3(T v) {
                x = y = z = v;
            }

            RENDER_CPU_GPU
            Point3(T xx, T yy, T zz) {
                x = xx;
                y = yy;
                z = zz;
            }

            RENDER_CPU_GPU
            T operator[](int i) const {
                CHECK(i >= 0 && i < 3);
                if (i == 0) {
                    return x;
                } else if (i == 1) {
                    return y;
                } else {
                    return z;
                }
            }

            RENDER_CPU_GPU
            T &operator[](int i) {
                CHECK(i >= 0 && i < 3);
                if (i == 0) {
                    return x;
                } else if (i == 1) {
                    return y;
                } else {
                    return z;
                }
            }

            RENDER_CPU_GPU
            T operator/(T v) const {
                assert(v != 0);
                return Point3<T>(x / v, y / v, z / v);
            }

            T x, y, z;
        };

        template<typename T>
        class Point4 {
        public:
            RENDER_CPU_GPU
            Point4() {
                x = y = z = w = 0;
            }

            RENDER_CPU_GPU
            Point4(T v) {
                x = y = z = w = v;
            }

            RENDER_CPU_GPU
            Point4(T xx, T yy, T zz, T ww) {
                x = xx;
                y = yy;
                z = zz;
                w = ww;
            }

            RENDER_CPU_GPU
            Point4(const Point3<T> &p, T v) {
                x = p.x;
                y = p.y;
                z = p.z;
                w = v;
            }

            RENDER_CPU_GPU
            T operator[](int i) const {
                CHECK(i >= 0 && i < 4);
                if (i == 0) {
                    return x;
                } else if (i == 1) {
                    return y;
                } else if (i == 2) {
                    return z;
                } else {
                    return w;
                }
            }

            RENDER_CPU_GPU
            T &operator[](int i) {
                CHECK(i >= 0 && i < 4);
                if (i == 0) {
                    return x;
                } else if (i == 1) {
                    return y;
                } else if (i == 2) {
                    return z;
                } else {
                    return w;
                }
            }

            T x, y, z, w;
        };

        template<typename T>
        class Normal3 {
        public:
            RENDER_CPU_GPU
            Normal3() {
                x = y = z = 0;
            }

            RENDER_CPU_GPU
            Normal3(T v) {
                x = y = z = v;
            }

            RENDER_CPU_GPU
            T operator[](int i) const {
                CHECK(i >= 0 && i < 3);
                if (i == 0) {
                    return x;
                } else if (i == 1) {
                    return y;
                } else {
                    return z;
                }
            }

            RENDER_CPU_GPU
            T &operator[](int i) {
                CHECK(i >= 0 && i < 3);
                if (i == 0) {
                    return x;
                } else if (i == 1) {
                    return y;
                } else {
                    return z;
                }
            }

            T x, y, z;
        };

        template<typename T>
        class Normal4 {
        public:
            RENDER_CPU_GPU
            Normal4() {
                x = y = z = w = 0;
            }

            RENDER_CPU_GPU
            Normal4(T v) {
                x = y = z = w = v;
            }

            RENDER_CPU_GPU
            Normal4(T xx, T yy, T zz, T ww) {
                x = xx;
                y = yy;
                z = zz;
                w = ww;
            }

            RENDER_CPU_GPU
            Normal4(const Normal3<T> &p, T v) {
                x = p.x;
                y = p.y;
                z = p.z;
                w = v;
            }

            RENDER_CPU_GPU
            T operator[](int i) const {
                CHECK(i >= 0 && i < 4);
                if (i == 0) {
                    return x;
                } else if (i == 1) {
                    return y;
                } else if (i == 2) {
                    return z;
                } else {
                    return w;
                }
            }

            RENDER_CPU_GPU
            T &operator[](int i) {
                CHECK(i >= 0 && i < 4);
                if (i == 0) {
                    return x;
                } else if (i == 1) {
                    return y;
                } else if (i == 2) {
                    return z;
                } else {
                    return w;
                }
            }

            T x, y, z, w;
        };

        template<typename T>
        class Matrix4 {
        public:
            RENDER_CPU_GPU
            Matrix4() {
                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 4; j++) {
                        _mat[i][j] = T(0);
                    }
                }
            }

            RENDER_CPU_GPU
            Matrix4(T v) {
                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 4; j++) {
                        _mat[i][j] = T(v);
                    }
                }
            }

            RENDER_CPU_GPU
            Matrix4(T mat[16]) {
                for (int row = 0; row < 4; row++) {
                    for (int col = 0; col < 4; col++) {
                        _mat[row][col] = *(mat + row * 4 + col);
                    }
                }
            }

            RENDER_CPU_GPU
            const T *operator[](size_t i) const {
                CHECK(i >= 0 && i < 4);
                size_t offset = i * 4;
                return _mat + offset;
            }

            RENDER_CPU_GPU
            T *operator[](size_t i) {
                CHECK(i >= 0 && i < 4);
                size_t offset = i * 4;
                return _mat + offset;
            }

            RENDER_CPU_GPU
            friend Point4<T> operator*(const Matrix4<T> m, const Point4<T> p) {
                T x = p.x, y = p.y, z = p.z, w = p.w;
                T v0 = m[0][0] * x + m[0][1] * y + m[0][2] * z + m[0][3] * w;
                T v1 = m[1][0] * x + m[1][1] * y + m[1][2] * z + m[1][3] * w;
                T v2 = m[2][0] * x + m[2][1] * y + m[2][2] * z + m[2][3] * w;
                T v3 = m[3][0] * x + m[3][1] * y + m[3][2] * z + m[3][3] * w;
                return Point4<T>(v0, v1, v2, v3);
            }

            RENDER_CPU_GPU
            friend Vector4<T> operator*(const Matrix4<T> m, const Vector4<T> p) {
                T x = p.x, y = p.y, z = p.z, w = p.w;
                T v0 = m[0][0] * x + m[0][1] * y + m[0][2] * z + m[0][3] * w;
                T v1 = m[1][0] * x + m[1][1] * y + m[1][2] * z + m[1][3] * w;
                T v2 = m[2][0] * x + m[2][1] * y + m[2][2] * z + m[2][3] * w;
                T v3 = m[3][0] * x + m[3][1] * y + m[3][2] * z + m[3][3] * w;
                return Vector4<T>(v0, v1, v2, v3);
            }

            RENDER_CPU_GPU
            friend Matrix4<T> inverse_transpose(const Matrix4<Float> m);

        private:
            T _mat[4][4];
        };

        using Vector3f = Vector3<Float>;
        using Vector3i = Vector3<int>;
        using Vector4f = Vector4<Float>;

        using Point4f = Point4<Float>;
        using Point4i = Point4<int>;

        using Point3f = Point3<Float>;
        using Point3i = Point3<int>;

        using Normal3f = Normal3<Float>;
        using Normal4f = Normal4<Float>;

        using Matrix4f = Matrix4<Float>;

        template<typename T>
        inline RENDER_CPU_GPU typename std::enable_if_t<std::is_floating_point<T>::value, bool> isNaN(T val) {
#ifdef __RENDER_GPU_MODE__
            return isnan(val);
#else
            return std::isnan(val);
#endif
        }
    }
}

#endif //TUNAN_MATH_H
