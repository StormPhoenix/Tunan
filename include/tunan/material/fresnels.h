//
// Created by Storm Phoenix on 2021/6/10.
//

#ifndef TUNAN_FRESNEL_H
#define TUNAN_FRESNEL_H

#include <tunan/common.h>
#include <tunan/math.h>

namespace RENDER_NAMESPACE {
    namespace fresnel {
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
    }
}

#endif //TUNAN_FRESNEL_H
