//
// Created by Storm Phoenix on 2021/6/10.
//

#ifndef TUNAN_FRESNEL_H
#define TUNAN_FRESNEL_H

#include <tunan/common.h>
#include <tunan/math.h>
#include <tunan/base/spectrum.h>

namespace RENDER_NAMESPACE {
    namespace fresnel {
        using base::Spectrum;

        RENDER_CPU_GPU
        inline Float fresnelDielectric(Float cosThetaI, Float etaI, Float etaT) {
            cosThetaI = math::clamp(cosThetaI, -1, 1);
            if (cosThetaI < 0) {
                // Traver form inner
                cosThetaI = std::abs(cosThetaI);
                Float tmp = etaI;
                etaI = etaT;
                etaT = tmp;
            }

            Float sinThetaI2 = std::max(Float(0.), Float(1 - cosThetaI * cosThetaI));
            Float sinThetaI = std::sqrt(sinThetaI2);
            Float sinThetaT = sinThetaI * (etaI / etaT);

            if (sinThetaT >= 1) {
                // Totally reflection
                return 1.0f;
            }

            Float cosThetaT2 = std::max(Float(0.), Float(1.0 - sinThetaT * sinThetaT));
            Float cosThetaT = std::sqrt(cosThetaT2);

            Float parallelR = ((etaT * cosThetaI) - (etaI * cosThetaT)) / ((etaT * cosThetaI) + (etaI * cosThetaT));
            Float perpendicularR = ((etaI * cosThetaI) - (etaT * cosThetaT)) /((etaI * cosThetaI) + (etaT * cosThetaT));
            return 0.5 * (parallelR * parallelR + perpendicularR * perpendicularR);
        }

        // References https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
        RENDER_CPU_GPU
        inline Spectrum fresnelConductor(Float cosThetaI, const Spectrum &etaI,
                                         const Spectrum &etaT, const Spectrum &k) {
            cosThetaI = math::clamp(cosThetaI, -1, 1);
            Spectrum eta = etaT / etaI;
            Spectrum K = k / etaI;

            Spectrum cosTheta2 = Spectrum(cosThetaI * cosThetaI);
            Spectrum sinTheta2 = Spectrum(1 - cosTheta2);

            Spectrum eta2 = eta * eta;
            Spectrum K2 = K * K;

            Spectrum item0 = Spectrum(eta2 - K2 - sinTheta2);
            Spectrum a2Andb2 = sqrt(item0 * item0 + 4 * eta2 * K2);
            Spectrum a = sqrt(0.5 * (a2Andb2 + item0));
            Spectrum item1 = a2Andb2 + cosTheta2;
            Spectrum item2 = 2. * a * cosThetaI;

            Spectrum perpendicularR = (item1 - item2) /
                                      (item1 + item2);

            Spectrum item3 = cosTheta2 * a2Andb2 + sinTheta2 * sinTheta2;
            Spectrum item4 = item2 * sinTheta2;

            Spectrum parallelR = perpendicularR * (item3 - item4) /
                                 (item3 + item4);
            return 0.5 * (perpendicularR + parallelR);
        }

        RENDER_CPU_GPU
        inline Spectrum fresnelSchlick(Float cosTheta, const Spectrum &R) {
            auto pow5Func = [](Float x) -> Float { return x * x * x * x * x; };
            return R + (Spectrum(1.) - R) * pow5Func(1 - cosTheta);
        }
    }
}

#endif //TUNAN_FRESNEL_H
