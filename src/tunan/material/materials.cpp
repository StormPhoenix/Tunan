//
// Created by Storm Phoenix on 2021/6/6.
//

#include <tunan/material/materials.h>
#include <tunan/material/bsdfs.h>

namespace RENDER_NAMESPACE {
    namespace material {
        using bsdf::LambertianBxDF;

        RENDER_CPU_GPU
        BSDF Lambertian::evaluateBSDF(SurfaceInteraction &si, LambertianBxDF *bxdf, TransportMode mode) {
            (*bxdf) = LambertianBxDF(_Kd.evaluate(si));
            BSDF bsdf = BSDF(si.ng, si.ns, si.wo);
            bsdf.setBxDF(bxdf);
            return bsdf;
        }

        Dielectric::Dielectric(SpectrumTexture R, SpectrumTexture T, Float etaI, Float etaT, Float roughness) :
                _R(R), _T(T), _etaI(etaI), _etaT(etaT), _roughness(roughness) {}

        RENDER_CPU_GPU
        bool Dielectric::isSpecular() const {
            return true;
            // TODO handle microfacets
//            return _roughness == 0.f;
        }

        RENDER_CPU_GPU
        BSDF Dielectric::evaluateBSDF(SurfaceInteraction &si, FresnelSpecularBxDF *bxdf, TransportMode mode) {
            BSDF bsdf = BSDF(si.ng, si.ns, si.wo);
            Spectrum Kr = _R.evaluate(si);
            Spectrum Kt = _T.evaluate(si);

            if (Kr.isBlack() && Kt.isBlack()) {
                return bsdf;
            }

            if (isSpecular()) {
                (*bxdf) = FresnelSpecularBxDF(Kr, Kt, _etaI, _etaT, mode);
                bsdf.setBxDF(bxdf);
                /* TODO handle microfacets
            } else {
                const GGXDistribution *distribution = ALLOC(memoryArena, GGXDistribution)(_roughness);
                insect.bsdf->addBXDF(ALLOC(memoryArena, BXDFMicrofacet)(Kr, Kt, _etaI, _etaT, distribution, mode));
                 */
            }
        }

        RENDER_CPU_GPU
        inline bool Material::isSpecular() {
            auto func = [&](auto ptr) { return ptr->isSpecular(); };
            return proxyCall(func);
        }
    }
}