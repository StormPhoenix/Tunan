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
                std::swap(etaI, etaT);
            }

            Float sinThetaI = std::sqrt((std::max)(Float(0.), Float(1 - std::pow(cosThetaI, 2))));
            Float sinThetaT = sinThetaI * (etaI / etaT);

            if (sinThetaT >= 1) {
                // Totally reflection
                return 1.0f;
            }

            Float cosThetaT = std::sqrt((std::max)(Float(0.), Float(1 - std::pow(sinThetaT, 2))));
            // 计算 R_parallel
            Float parallelR = ((etaT * cosThetaI) - (etaI * cosThetaT)) /
                              ((etaT * cosThetaI) + (etaI * cosThetaT));
            Float perpendicularR = ((etaI * cosThetaI) - (etaT * cosThetaT)) /
                                   ((etaI * cosThetaI) + (etaT * cosThetaT));
            return 0.5 * (parallelR * parallelR + perpendicularR * perpendicularR);
        }

    }
}

#endif //TUNAN_FRESNEL_H
