//
// Created by Storm Phoenix on 2021/6/20.
//

#include <tunan/medium/phase_functions.h>

namespace RENDER_NAMESPACE {
    RENDER_CPU_GPU
    static Float henyeyGreensteinPdf(Float cosTheta, Float g) {
        Float denominator = (1 + g * g + 2 * g * cosTheta);
        denominator = denominator * std::sqrt(denominator);
        return Inv_4Pi * (1 - g * g) / denominator;
    }

    RENDER_CPU_GPU
    HGFunc::HGFunc(Float g) : _G(g) {}

    RENDER_CPU_GPU
    Float HGFunc::sample(const Vector3F &wo, Vector3F *wi, Vector2F sample) const {
        // Henyey-Greenstein phase function
        Float sampleU = sample.x;
        Float sampleV = sample.y;

        // \cos(\theta) \sin(\theta)
        Float term1 = (1 - _G * _G) / (1 + _G * (2 * sampleU - 1));
        Float cosTheta;
        if (std::abs(_G) < Epsilon) {
            cosTheta = 1 - 2 * sampleU;
        } else {
            cosTheta = (1 + _G * _G - term1 * term1) / (2 * _G);
        }
        Float sinTheta = std::sqrt(std::max(Float(0.), 1 - cosTheta * cosTheta));

        // \phi
        Float phi = 2 * Pi * sampleV;
        Float cosPhi = std::cos(phi);
        Float sinPhi = std::sin(phi);

        // build local coordinate space
        Vector3F tanY = NORMALIZE(wo);
        Vector3F tanX;
        Vector3F tanZ;
        math::tangentSpace(tanY, &tanX, &tanZ);

        // calculate coordinate
        Float y = cosTheta;
        Float x = sinTheta * cosPhi;
        Float z = sinTheta * sinPhi;

        (*wi) = NORMALIZE(x * tanX + y * tanY + z * tanZ);
        return henyeyGreensteinPdf(cosTheta, _G);
    }

    RENDER_CPU_GPU
    Float HGFunc::pdf(const Vector3F &wo, const Vector3F &wi) const {
        return henyeyGreensteinPdf(DOT(wo, wi), _G);
    }
}