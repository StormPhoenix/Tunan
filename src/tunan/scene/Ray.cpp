//
// Created by StormPhoenix on 2021/6/2.
//

#include <tunan/scene/Ray.h>

namespace RENDER_NAMESPACE {
    using namespace math;
    RENDER_CPU_GPU
    Ray::Ray() : _minStep(0.0), _step(Infinity) {}

    RENDER_CPU_GPU
    Ray::Ray(const Vector3F &origin, const Vector3F &direction) :
            _origin(origin), _direction(direction), _minStep(0.0), _step(Infinity) {}

    RENDER_CPU_GPU
    const Vector3F &Ray::getDirection() const {
        return _direction;
    }

    RENDER_CPU_GPU
    const Vector3F &Ray::getOrigin() const {
        return _origin;
    }

    RENDER_CPU_GPU
    void Ray::setStep(Float step) {
        if (step < _minStep) {
            _step = _minStep;
            return;
        }
        _step = step;
    }

    RENDER_CPU_GPU
    const Vector3F Ray::at(Float step) const {
        return _origin + Vector3F(_direction.x * step, _direction.y * step, _direction.z * step);
    }
}