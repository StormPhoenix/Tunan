//
// Created by StormPhoenix on 2021/6/2.
//

#ifndef TUNAN_RAY_H
#define TUNAN_RAY_H

#include <tunan/common.h>
#include <tunan/math.h>
#include <tunan/medium/mediums.h>

namespace RENDER_NAMESPACE {
    class Ray {
    public:
        RENDER_CPU_GPU
        Ray();

        RENDER_CPU_GPU
        Ray(const Vector3F &origin, const Vector3F &direction);

        RENDER_CPU_GPU
        Ray(const Vector3F &origin, const Vector3F &direction, Medium medium);

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
        Point3F at(Float step) const;

        RENDER_CPU_GPU
        Medium getMedium() const {
            return _medium;
        }

        RENDER_CPU_GPU
        void setMedium(Medium medium) {
            _medium = medium;
        }

    private:
        Vector3F _origin;
        Vector3F _direction;
        Float _minStep;
        Float _step;
        Medium _medium;
    };

}

#endif //TUNAN_RAY_H
