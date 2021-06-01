//
// Created by StormPhoenix on 2021/5/31.
//

#ifndef TUNAN_MATH_H
#define TUNAN_MATH_H

#include <tunan/common.h>

#ifdef __RENDER_GPU_MODE__
#include <cuda.h>
#endif

#include <ext/glm/glm.hpp>
#include <ext/glm/gtc/matrix_transform.hpp>
#include <ext/glm/gtc/matrix_inverse.hpp>
#include <ext/glm/gtx/compatibility.hpp>

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

using Point2F = Vector2F;
using Point3F = Vector3F;
using Point4F = Vector4F;
using Normal3F = Vector3F;
using Point2I = Vector2i;
using Point3I = Vector3i;
using Point4I = Vector4i;

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
    namespace math {
        template<typename T>
        inline RENDER_CPU_GPU typename std::enable_if_t<std::is_floating_point<T>::value, bool> isNaN(T val) {
#ifdef __RENDER_GPU_MODE__
            return isnan(val);
#else
            return std::isnan(v);
#endif
        }
    }
}

#endif //TUNAN_MATH_H
