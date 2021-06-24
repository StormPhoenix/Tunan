//
// Created by Storm Phoenix on 2021/6/8.
//

#include <tunan/scene/lights.h>
#include <tunan/utils/ResourceManager.h>
#include <tunan/utils/image_utils.h>

#include <string>
#include <fstream>

namespace RENDER_NAMESPACE {
    using namespace utils;

    SpotLight::SpotLight(const Spectrum &intensity, Transform lightToWorld, MediumInterface mediumBoundary,
                         Float fallOffRange, Float totalRange) :
            _type(Delta_Position), _mediumInterface(mediumBoundary), _intensity(intensity),
            _lightToWorld(lightToWorld) {
        _center = _lightToWorld.transformPoint(Point3F(0));
        _direction = NORMALIZE(_lightToWorld.transformVector(Vector3F(0, 1, 0)));

        _cosFallOffRange = std::cos(math::degreesToRadians(fallOffRange));
        _cosTotalRange = std::cos(math::degreesToRadians(totalRange));
    }

    RENDER_CPU_GPU
    Spectrum SpotLight::sampleLi(const Interaction &eye, Vector3F *wi, Float *pdf,
                                 Vector2F uv, Interaction *target) {
        (*wi) = NORMALIZE(_center - eye.p);
        (*pdf) = 1.0;

        if (target != nullptr) {
            target->p = _center;
            target->ng = -(*wi);
            target->wo = -(*wi);
            target->mediumInterface = _mediumInterface;
            target->error = Vector3F(0);
        }
        Float distance = LENGTH(_center - eye.p);
        return _intensity * fallOffWeight(-(*wi)) / (distance * distance);
    }

    RENDER_CPU_GPU
    Float SpotLight::pdfLi(const Interaction &eye, const Vector3F &direction) {
        return 0;
    }

    RENDER_CPU_GPU
    Spectrum SpotLight::fallOffWeight(const Vector3F &wo) {
        Float cosine = DOT(wo, _direction);
        if (cosine < _cosTotalRange) {
            return Spectrum(0);
        } else if (cosine < _cosFallOffRange) {
            Float fraction = (cosine - _cosTotalRange) / (_cosFallOffRange - _cosTotalRange);
            return Spectrum(fraction * fraction * fraction * fraction);
        } else {
            return Spectrum(1.f);
        }
    }

    DiffuseAreaLight::DiffuseAreaLight(const Spectrum &radiance, Shape shape,
                                       MediumInterface mediumInterface, bool twoSided) :
            _radiance(radiance), _shape(shape), _type(Area), _mediumInterface(mediumInterface), _twoSided(twoSided) {}

    RENDER_CPU_GPU
    Spectrum DiffuseAreaLight::sampleLi(const Interaction &eye, Vector3F *wi, Float *pdf,
                                        Vector2F uv, Interaction *target) {
        assert(!_shape.nullable());
        SurfaceInteraction si = _shape.sample(eye.p, pdf, uv);
        (*wi) = NORMALIZE(si.p - eye.p);
        if (target != nullptr) {
            target->p = si.p;
            target->ng = si.ng;
            target->wo = -(*wi);
            target->error = si.error;
            target->mediumInterface = _mediumInterface;
        }
        return L(si, -(*wi));
    }

    RENDER_CPU_GPU
    Spectrum DiffuseAreaLight::L(const Interaction &interaction, const Vector3F &wo) const {
        Float cosTheta = DOT(interaction.ng, wo);
        return (_twoSided || cosTheta > 0) ? _radiance : Spectrum(0.0);
    }

    RENDER_CPU_GPU
    Float DiffuseAreaLight::pdfLi(const Interaction &eye, const Vector3F &dir) {
        assert(_shape != nullptr);
        return _shape.surfaceInteractionPdf(eye.p, dir);
    }

    RENDER_CPU_GPU
    LightSourceType DiffuseAreaLight::getType() const {
        return _type;
    }

    PointLight::PointLight(const Spectrum &intensity, Transform lightToWorld, MediumInterface mediumInterface) :
            _type(Delta_Position), _mediumInterface(mediumInterface), _intensity(intensity),
            _lightToWorld(lightToWorld) {
        _center = lightToWorld.transformPoint(Point3F(0));
    }

    RENDER_CPU_GPU
    Spectrum PointLight::sampleLi(const Interaction &eye, Vector3F *wi, Float *pdf, Vector2F uv, Interaction *target) {
        (*wi) = NORMALIZE(_center - eye.p);
        (*pdf) = 1;

        if (target != nullptr) {
            target->p = _center;
            target->ng = -(*wi);
            target->wo = -(*wi);
            target->error = Vector3F(0);
            target->mediumInterface = _mediumInterface;
        }

        Float distance = LENGTH(_center - eye.p);
        return _intensity / (distance * distance);
    }

    RENDER_CPU_GPU
    Float PointLight::pdfLi(const Interaction &eye, const Vector3F &dir) {
        return 0.0;
    }

    EnvironmentLight::EnvironmentLight(Float intensity, std::string texturePath, MediumInterface mediumInterface,
                                       Transform lightToWorld, utils::ResourceManager *allocator) :
            _type(Environment), _intensity(intensity), _mediumInterface(mediumInterface), _lightToWorld(lightToWorld) {
        // Check file exists
        {
            std::ifstream in(texturePath);
            ASSERT(in.good(), texturePath + " NOT EXISTS.");
        }

        // Copy from image to texture
        _worldToLight = _lightToWorld.inverse();
        int channelsInFile = 3;
        {
            float *image = readImage(texturePath, &_width, &_height, SpectrumChannel, &channelsInFile);
            _texture = allocator->allocateObjects<Spectrum>(_width * _height);
            for (int row = 0; row < _height; row++) {
                for (int col = 0; col < _width; col++) {
                    int imageOffset = (row * _width + col) * channelsInFile;
                    int textureOffset = (row * _width + col);

                    for (int ch = 0; ch < SpectrumChannel; ch++) {
                        (*(_texture + textureOffset))[ch] = *(image + imageOffset + ch);
                    }
                }
            }
            delete image;
        }

        Float *sampleFunction = new Float[_width * _height];
        int sampleChannel = 1;
        for (int row = 0; row < _height; row++) {
            // SinTheta for sampling correction
            Float sinTheta = std::sin(Float(_height - 1 - row) / (_height - 1) * Pi);
            for (int col = 0; col < _width; col++) {
                int offset = row * _width + col;
                sampleFunction[offset] = _texture[offset][sampleChannel] * sinTheta;
            }
        }
        _textureDistribution = Distribution2D(sampleFunction, _width, _height, allocator);
    }

    RENDER_CPU_GPU
    Spectrum EnvironmentLight::sampleLi(const Interaction &eye, Vector3F *wi, Float *pdf,
                                        Vector2F uv, Interaction *target) {
        // Sample from environment map
        Float samplePdf = 0.;
        Point2F sample = _textureDistribution.sampleContinuous(&samplePdf, uv);

        if (samplePdf == 0.) {
            return Spectrum(0.);
        }

        // Inverse coordinate v
        Float theta = Pi * (1.0 - sample.y);
        Float phi = 2 * Pi * sample.x;

        Float sinTheta = std::sin(theta);
        Float cosTheta = std::cos(theta);
        Float cosPhi = std::cos(phi);
        Float sinPhi = std::sin(phi);

        *wi = _lightToWorld.transformVector({sinTheta * cosPhi, cosTheta, sinTheta * sinPhi});
        *wi = NORMALIZE(*wi);

        if (sinTheta == 0) {
            *pdf = 0;
        } else {
            // Jacobi correction
            *pdf = samplePdf / (2 * Pi * Pi * sinTheta);
        }

        if (target != nullptr) {
            target->p = eye.p + (*wi) * Float(2.f * _worldRadius);
            target->ng = -(*wi);
            target->wo = -(*wi);
            target->mediumInterface = _mediumInterface;
            target->error = Vector3F(0);
        }
        return sampleTexture(sample);
    }

    RENDER_CPU_GPU
    Float EnvironmentLight::pdfLi(const Interaction &eye, const Vector3F &dir) {
        Vector3F wi = NORMALIZE(_worldToLight.transformVector(dir));
        Float theta = math::local_coord::vectorTheta(wi);
        Float phi = math::local_coord::vectorPhi(wi);

        Float u = phi * Inv_2Pi;
        // Inverse the v coordinator
        Float v = (1.0 - theta * Inv_Pi);

        Float sinTheta = std::sin(theta);
        if (sinTheta == 0) {
            return 0;
        } else {
            // Jacobi correction
            return (_textureDistribution.pdfContinuous(Point2F(u, v))) / (2 * Pi * Pi * sinTheta);
        }
    }

    RENDER_CPU_GPU
    Spectrum EnvironmentLight::Le(const Ray &ray) const {
        Vector3F wi = NORMALIZE(_worldToLight.transformVector(ray.getDirection()));
        // wi.y = cosTheta
        Float theta = math::local_coord::vectorTheta(wi);
        // wi.x = sinTheta * cosPhi
        // wi.z = sinTheta * sinPhi
        // wi.z / wi.x = tanPhi
        Float phi = math::local_coord::vectorPhi(wi);
        // Inverse coordinator v
        Point2F uv = {phi * Inv_2Pi, (1.0 - theta * Inv_Pi)};
        return sampleTexture(uv);
    }

    void EnvironmentLight::worldBound(Point3F &worldMin, Point3F &worldMax) {
        _worldRadius = 0.5 * LENGTH(worldMax - worldMin);
        _worldCenter = (worldMax + worldMin) / Float(2.0);
    }

    RENDER_CPU_GPU
    Spectrum EnvironmentLight::sampleTexture(Point2F uv) const {
        int wOffset, hOffset;
        wOffset = uv[0] * _width;
        hOffset = uv[1] * _height;

        if (wOffset < 0 || wOffset >= _width
            || hOffset < 0 || hOffset >= _height) {
            return Spectrum(0);
        }

        // flip
//            hOffset = _height - (hOffset + 1);
        int offset = (hOffset * _width + wOffset);

        Spectrum ret(0);
        // TODO Adjust rgb channels
        for (int ch = 0; ch < 3 && ch < SpectrumChannel; ch++) {
//                ret[ch] = Float(_texture[offset][ch]) / 255.0 * _intensity;
            ret[ch] = Float(_texture[offset][ch]) * _intensity;
        }
        return ret;
    }

    RENDER_CPU_GPU
    LightSourceType EnvironmentLight::getType() const {
        return _type;
    }

    SunLight::SunLight(const Spectrum &intensity, const Vector3F &direction) :
            _type(Delta_Direction), L(intensity), _direction(direction), _mediumInterface(MediumInterface()) {}

    RENDER_CPU_GPU
    Spectrum SunLight::sampleLi(const Interaction &eye, Vector3F *wi, Float *pdf, Vector2F uv, Interaction *target) {
        if (wi != nullptr) {
            (*wi) = -_direction;
        }

        if (pdf != nullptr) {
            (*pdf) = 1.0;
        }

        if (target != nullptr) {
            target->p = eye.p + (-_direction * Float(2.0 * _worldRadius));
            target->ng = -(*wi);
            target->wo = -(*wi);
            target->mediumInterface = _mediumInterface;
            target->error = Vector3F(0);
        }
        return L;
    }

    RENDER_CPU_GPU
    Float SunLight::pdfLi(const Interaction &eye, const Vector3F &dir) {
        return 0.0f;
    }

    RENDER_CPU_GPU
    LightSourceType SunLight::getType() const {
        return _type;
    }

    void SunLight::worldBound(Point3F &worldMin, Point3F &worldMax) {
        _worldRadius = 0.5 * LENGTH(worldMax - worldMin);
        _worldCenter = (worldMax + worldMin) / Float(2.0);
    }

    RENDER_CPU_GPU
    Spectrum Light::sampleLi(const Interaction &eye, Vector3F *wi, Float *pdf, Vector2F uv, Interaction *target) {
        auto func = [&](auto ptr) {
            return ptr->sampleLi(eye, wi, pdf, uv, target);
        };
        return proxyCall(func);
    }

    RENDER_CPU_GPU
    Float Light::pdfLi(const Interaction &eye, const Vector3F &direction) {
        auto func = [&](auto ptr) {
            return ptr->pdfLi(eye, direction);
        };
        return proxyCall(func);
    }

    RENDER_CPU_GPU
    Spectrum Light::Le(const Ray &ray) {
        auto func = [&](auto ptr) {
            return ptr->Le(ray);
        };
        return proxyCall(func);
    }

    RENDER_CPU_GPU
    LightSourceType Light::getType() const {
        auto func = [&](auto ptr) {
            return ptr->getType();
        };
        return proxyCall(func);
    }

    RENDER_CPU_GPU
    bool Light::isDeltaType() const {
        auto func = [&](auto ptr) {
            LightSourceType type = ptr->getType();
            return (type & (Delta_Direction | Delta_Position)) > 0;
        };
        return proxyCall(func);
    }
}