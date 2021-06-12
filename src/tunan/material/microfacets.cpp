//
// Created by Storm Phoenix on 2021/6/11.
//

#include <tunan/material/microfacets.h>

namespace RENDER_NAMESPACE {
    namespace microfacet {
        RENDER_CPU_GPU
        GGXDistribution::GGXDistribution(Float alpha_g) : _alpha_g(alpha_g) {
            assert(_alpha_g > 0);
        }

        RENDER_CPU_GPU
        Float GGXDistribution::D(const Normal3F &wh) const {
            Float cosThetaH = math::local_coord::vectorCosTheta(wh);
            // ignore wh direction eg. if (cosThetaH < 0) -> return zero.
            if (cosThetaH == 0) {
                return 0;
            }


            Float tanTheta2Wh = math::local_coord::vectorTanTheta2(wh);
            if (math::isInf(tanTheta2Wh)) {
                return 0.;
            }

            Float cosThetaH4 = cosThetaH * cosThetaH * cosThetaH * cosThetaH;
            Float item = _alpha_g * _alpha_g + tanTheta2Wh;
            return (_alpha_g * _alpha_g) / (Pi * cosThetaH4 * item * item);
        }

        RENDER_CPU_GPU
        Float GGXDistribution::G(const Vector3F &v, const Normal3F &wh) const {
            if (DOT(v, wh) * DOT(v, Vector3F(0, 1, 0)) < 0.) {
                return 0.;
            }

            Float tanTheta2V = math::local_coord::vectorTanTheta2(v);
            if (math::isInf(tanTheta2V)) {
                return 0.;
            }

            if (tanTheta2V == 0.0f) {
                return 1.0f;
            }

            Float alpha2 = _alpha_g * _alpha_g;
            return 2.0 / (1 + std::sqrt(1 + alpha2 * tanTheta2V));
        }

        RENDER_CPU_GPU
        Float GGXDistribution::G(const Vector3F &wo, const Vector3F &wi, const Normal3F &wh) const {
            return G(wo, wh) * G(wi, wh);
        }

        RENDER_CPU_GPU
        Vector3F GGXDistribution::sampleWh(const Vector3F &wo, Vector2F &uv) const {
            Float phi = 2 * Pi * uv[0];
            Float theta = std::atan((_alpha_g * sqrt(uv[1])) / (std::sqrt(1 - uv[1])));

            Float cosTheta = std::cos(theta);
            Float sinTheta = std::sin(theta);
            Float cosPhi = std::cos(phi);
            Float sinPhi = std::sin(phi);
            Vector3F wh = Vector3F(sinTheta * cosPhi, cosTheta, sinTheta * sinPhi);
            // Correct wh to the same hemi-sphere
            if (wo.y * wh.y < 0) {
                wh *= -1;
            }
            return NORMALIZE(wh);
        }

        RENDER_CPU_GPU
        Float MicrofacetDistribution::D(const Normal3F &wh) const {
            auto func = [&](auto ptr) {
                return ptr->D(wh);
            };
            return proxyCall(func);
        }

        RENDER_CPU_GPU
        Float MicrofacetDistribution::G(const Vector3F &v, const Normal3F &wh) const {
            auto func = [&](auto ptr) {
                return ptr->G(v, wh);
            };
            return proxyCall(func);
        }

        RENDER_CPU_GPU
        Float MicrofacetDistribution::G(const Vector3F &wo, const Vector3F &wi, const Normal3F &wh) const {
            auto func = [&](auto ptr) {
                return ptr->G(wo, wi, wh);
            };
            return proxyCall(func);
        }

        RENDER_CPU_GPU
        Vector3F MicrofacetDistribution::sampleWh(const Vector3F &wo, Vector2F &uv) const {
            auto func = [&](auto ptr) {
                return ptr->sampleWh(wo, uv);
            };
            return proxyCall(func);
        }

        RENDER_CPU_GPU
        Float MicrofacetDistribution::samplePdf(const Vector3F &wo, const Vector3F &wh) const {
            // cos(theta) = shift (D(w_h) * G(w_h, w) * cos(w, w_h)) d_wh
//                return D(wh) * G(wo) * ABS_DOT(wo, wh) / std::abs(wo.y);
            return D(wh) * std::abs(math::local_coord::vectorCosTheta(wh));
        }
    }
}