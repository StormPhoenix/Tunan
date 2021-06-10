//
// Created by Storm Phoenix on 2021/6/10.
//

#ifndef TUNAN_MAPPINGS_H
#define TUNAN_MAPPINGS_H

#include <tunan/math.h>
#include <tunan/common.h>
#include <tunan/base/transform.h>
#include <tunan/base/interactions.h>
#include <tunan/utils/TaggedPointer.h>
#include <tunan/utils/MemoryAllocator.h>

namespace RENDER_NAMESPACE {
    namespace material {
        using utils::TaggedPointer;
        using utils::MemoryAllocator;
        using base::Transform;
        using base::SurfaceInteraction;

        class UVMapping2D {
        public:
            UVMapping2D(Float uScale = 1.0, Float vScale = 1.0) :
                    _uScale(uScale), _vScale(vScale) {}

            RENDER_CPU_GPU
            Point2F map(const SurfaceInteraction &si) {
                return Point2F(si.uv[0] * _uScale, si.uv[1] * _vScale);
            }

        private:
            Float _uScale;
            Float _vScale;
        };

        class SphericalMapping2D {
        public:
            SphericalMapping2D(Transform modelViewMatrix)
                    : _modelViewMatrix(modelViewMatrix) {
                _worldToLocalMatrix = _modelViewMatrix.inverse();
            }

            RENDER_CPU_GPU
            Point2F map(const SurfaceInteraction &si) {
                Point3F dir = NORMALIZE(_worldToLocalMatrix.transformPoint(si.p) - Point3F(0., 0., 0.));

                Float theta = math::local_coord::vectorTheta(dir);
                Float phi = math::local_coord::vectorPhi(dir);
                return Point2F(theta * Inv_Pi, phi * Inv_2Pi);
            }

        private:
            Transform _modelViewMatrix;
            Transform _worldToLocalMatrix;
        };

        class TextureMapping2D : public TaggedPointer<UVMapping2D, SphericalMapping2D> {
        public:
            using TaggedPointer::TaggedPointer;

            RENDER_CPU_GPU
            Point2F map(const SurfaceInteraction &si);
        };
    }
}

#endif //TUNAN_MAPPINGS_H
