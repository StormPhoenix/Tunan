//
// Created by StormPhoenix on 2021/6/2.
//

#ifndef TUNAN_RAY_H
#define TUNAN_RAY_H

#include <tunan/common.h>
#include <tunan/base/math.h>

namespace RENDER_NAMESPACE {

    class Ray {
    public:
        Ray();

        RENDER_CPU_GPU
        Ray(const Vector3F &origin, const Vector3F &direction);

        RENDER_CPU_GPU
        const Vector3F &getDirection() const;

        RENDER_CPU_GPU
        const Point3F &getOrigin() const;

        RENDER_CPU_GPU
        Float getMinStep() const {
            return _minStep;
        }

        RENDER_CPU_GPU
        Float getStep() const {
            return _step;
        }

        RENDER_CPU_GPU
        void setStep(Float step);

        RENDER_CPU_GPU
        const Vector3F at(Float step) const;

    private:
        Vector3F _origin;
        Vector3F _direction;
        Float _minStep;
        Float _step;
    };

}

#endif //TUNAN_RAY_H
