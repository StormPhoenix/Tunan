//
// Created by Storm Phoenix on 2021/6/6.
//

#include <tunan/material/bsdfs.h>
#include <tunan/material/materials.h>
#include <tunan/material/microfacets.h>

namespace RENDER_NAMESPACE {
    namespace material {
        using bsdf::LambertianBxDF;
        using microfacet::GGXDistribution;

        RENDER_CPU_GPU
        BSDF Lambertian::evaluateBSDF(SurfaceInteraction &si, LambertianBxDF *bxdf, TransportMode mode) {
            (*bxdf) = LambertianBxDF(_Kd.evaluate(si));
            BSDF bsdf = BSDF(si.ng, si.ns, si.wo);
            bsdf.setBxDF(bxdf);
            return bsdf;
        }

        Dielectric::Dielectric(SpectrumTexture R, SpectrumTexture T, Float etaI, Float etaT, Float roughness) :
                _R(R), _T(T), _etaI(etaI), _etaT(etaT), _roughness(roughness) {
            ASSERT(etaI != 0.f && etaT != 0.f, "Eta can't be zero. ");
        }

        RENDER_CPU_GPU
        BSDF Dielectric::evaluateBSDF(SurfaceInteraction &si, DielectricBxDF *bxdf, TransportMode mode) {
            Spectrum Kr = _R.evaluate(si);
            Spectrum Kt = _T.evaluate(si);
            Float roughness = _roughness;

            BSDF bsdf = BSDF(si.ng, si.ns, si.wo);
            if (Kr.isBlack() && Kt.isBlack()) {
                return bsdf;
            }

            (*bxdf) = DielectricBxDF(Kr, Kt, roughness, _etaI, _etaT, GGX, mode);
            bsdf.setBxDF(bxdf);
            bsdf.refractionIndex = DOT(si.ng, si.wo) > 0 ? _etaI / _etaT : _etaT / _etaI;
            return bsdf;
        }

        Mirror::Mirror() {}

        Mirror::Mirror(SpectrumTexture &Ks) : _Ks(Ks) {}

        RENDER_CPU_GPU
        BSDF Mirror::evaluateBSDF(SurfaceInteraction &si, SpecularReflectionBxDF *bxdf, TransportMode mode) {
            BSDF bsdf = BSDF(si.ng, si.ns, si.wo);
            if (_Ks.nullable()) {
                (*bxdf) = SpecularReflectionBxDF(Spectrum(1.0));
            } else {
                (*bxdf) = SpecularReflectionBxDF(_Ks.evaluate(si));
            }
            bsdf.setBxDF(bxdf);
            return bsdf;
        }

        Metal::Metal(FloatTexture alpha, SpectrumTexture eta, SpectrumTexture Ks, SpectrumTexture K,
                     MicrofacetDistribType distribType) :
                _alpha(alpha), _eta(eta), _Ks(Ks), _K(K), _distribType(distribType) {
            ASSERT(!_alpha.nullable(), "Alpha is nullptr. ");
            ASSERT(!_eta.nullable(), "Eta is nullptr. ");
            ASSERT(!_Ks.nullable(), "R is nullptr. ");
            ASSERT(!_K.nullable(), "K is nullptr. ");
        }

        RENDER_CPU_GPU
        BSDF Metal::evaluateBSDF(SurfaceInteraction &si, ConductorBxDF *bxdf, TransportMode mode) {
            Float alpha = _alpha.evaluate(si);
            Spectrum Ks = _Ks.evaluate(si);
            Spectrum etaT = _eta.evaluate(si);
            Spectrum K = _K.evaluate(si);

            BSDF bsdf = BSDF(si.ng, si.ns, si.wo);
            new(bxdf) ConductorBxDF(Ks, Spectrum(1.f), etaT, K, alpha, _distribType);

            bsdf.setBxDF(bxdf);
            return bsdf;
        }

        Patina::Patina(SpectrumTexture Kd, SpectrumTexture Ks, FloatTexture alpha, MicrofacetDistribType distribType) :
                _Kd(Kd), _Ks(Ks), _alpha(alpha), _distribType(distribType) {}

        RENDER_CPU_GPU
        BSDF Patina::evaluateBSDF(SurfaceInteraction &si, GlossyDiffuseBxDF *bxdf, TransportMode mode) {
            Float alpha = _alpha.evaluate(si);
            Spectrum Kd = _Kd.evaluate(si);
            Spectrum Ks = _Ks.evaluate(si);

            BSDF bsdf = BSDF(si.ng, si.ns, si.wo);
            new(bxdf) GlossyDiffuseBxDF(Kd, Ks, alpha, _distribType);

            bsdf.setBxDF(bxdf);
            return bsdf;
        }
    }
}