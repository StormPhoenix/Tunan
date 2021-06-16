//
// Created by StormPhoenix on 2021/6/7.
//

#include <tunan/math.h>
#include <tunan/material/bsdfs.h>
#include <tunan/material/fresnels.h>
#include <tunan/sampler/samplers.h>

namespace RENDER_NAMESPACE {
    namespace bsdf {
        RENDER_CPU_GPU
        LambertianBxDF::LambertianBxDF() :
                _type(BxDFType(BSDF_Diffuse | BSDF_Reflection)), _Kd(0) {}

        RENDER_CPU_GPU
        LambertianBxDF::LambertianBxDF(const Spectrum &Kd) :
                _type(BxDFType(BSDF_Diffuse | BSDF_Reflection)), _Kd(Kd) {}

        RENDER_CPU_GPU
        Spectrum LambertianBxDF::f(const Vector3F &wo, const Vector3F &wi) const {
            return _Kd * Inv_Pi;
        }

        RENDER_CPU_GPU
        Spectrum LambertianBxDF::sampleF(const Vector3F &wo, Vector3F *wi, Float *pdf,
                                         BSDFSample &bsdfSample, BxDFType *sampleType) {
            *wi = sampler::hemiCosineSampling(bsdfSample.uv);
            if (wo.y < 0) {
                wi->y *= -1;
            }
            *pdf = samplePdf(wo, *wi);
            if (sampleType != nullptr) {
                (*sampleType) = BxDFType(BSDF_Diffuse | BSDF_Reflection);
            }
            return f(wo, *wi);
        }

        RENDER_CPU_GPU
        Float LambertianBxDF::samplePdf(const Vector3F &wo, const Vector3F &wi) const {
            if (wo.y * wi.y > 0) {
                return std::abs(wi.y) * Inv_Pi;
            } else {
                return 0;
            }
        }

        RENDER_CPU_GPU
        FresnelSpecularBxDF::FresnelSpecularBxDF() :
                _type(BxDFType(BSDF_Specular | BSDF_Reflection | BSDF_Transmission)) {}

        RENDER_CPU_GPU
        FresnelSpecularBxDF::FresnelSpecularBxDF(const Spectrum &reflectance, const Spectrum &transmittance,
                                                 Float thetaI, Float thetaT, TransportMode mode) :
                _type(BxDFType(BSDF_Specular | BSDF_Reflection | BSDF_Transmission)),
                _reflectance(reflectance), _transmittance(transmittance),
                _thetaI(thetaI), _thetaT(thetaT), _mode(mode) {}

        RENDER_CPU_GPU
        Spectrum FresnelSpecularBxDF::f(const Vector3F &wo, const Vector3F &wi) const {
            return Spectrum(0.0);
        }

        RENDER_CPU_GPU
        Spectrum FresnelSpecularBxDF::sampleF(const Vector3F &wo, Vector3F *wi, Float *pdf,
                                              BSDFSample &bsdfSample, BxDFType *sampleType) {
            Float cosine = wo.y;
            // compute reflect probability
            Float reflectProb = fresnel::fresnelDielectric(cosine, _thetaI, _thetaT);
            // reflect probability approximation
//            Float reflectProb = math::schlick(cosine, _thetaI / _thetaT);

            Float random = bsdfSample.uv[0];
            if (random < reflectProb) {
                // do reflection
                if (sampleType != nullptr) {
                    *sampleType = BxDFType(BSDF_Specular | BSDF_Reflection);
                }
                *wi = Vector3F(-wo.x, wo.y, -wo.z);
                *pdf = reflectProb;
                // f(p, w_o, w_i) / cos(theta(w_i))
                if ((*wi).y == 0) {
                    return Spectrum(0);
                }
                return reflectProb * _reflectance / abs(wi->y);
            } else {
                // refraction
                Vector3F normal = Vector3F(0.0, 1.0, 0.0);
                Float refraction;
                if (wo.y > 0) {
                    // Travel from outer
                    refraction = _thetaI / _thetaT;
                } else {
                    // Travel from inner
                    refraction = _thetaT / _thetaI;
                    normal.y *= -1;
                }

                if (!math::refract(wo, normal, refraction, wi)) {
                    // Totally reflection
                    return Spectrum(0);
                }

                *pdf = 1 - reflectProb;
                Spectrum f = _transmittance * (1.0 - reflectProb) / std::abs(wi->y);
                if (_mode == RADIANCE) {
                    f *= (refraction * refraction);
                }
                if (sampleType != nullptr) {
                    *sampleType = BxDFType(BSDF_Specular | BSDF_Transmission);
                }
                return f;
            }
        }

        RENDER_CPU_GPU
        Float FresnelSpecularBxDF::samplePdf(const Vector3F &wo, const Vector3F &wi) const {
            return 0.0;
        }

        RENDER_CPU_GPU
        SpecularReflectionBxDF::SpecularReflectionBxDF() :
                _type(BxDFType(BSDF_Specular | BSDF_Reflection)) {}

        RENDER_CPU_GPU
        SpecularReflectionBxDF::SpecularReflectionBxDF(const Spectrum &Ks) :
                _type(BxDFType(BSDF_Specular | BSDF_Reflection)),
                _Ks(Ks) {}

        RENDER_CPU_GPU
        Spectrum SpecularReflectionBxDF::f(const Vector3F &wo, const Vector3F &wi) const {
            return Spectrum(0.0f);
        }

        RENDER_CPU_GPU
        Spectrum SpecularReflectionBxDF::sampleF(const Vector3F &wo, Vector3F *wi, Float *pdf,
                                                 BSDFSample &bsdfSample, BxDFType *sampleType) {
            *wi = Vector3F(-wo.x, wo.y, -wo.z);
            *pdf = 1;
            if (sampleType != nullptr) {
                *sampleType = BxDFType(BSDF_Specular | BSDF_Reflection);
            }

            Float cosTheta = std::abs((*wi).y);
            if (cosTheta == 0.) {
                return Spectrum(0.);
            }
            return _Ks / cosTheta;
        }

        RENDER_CPU_GPU
        Float SpecularReflectionBxDF::samplePdf(const Vector3F &wo, const Vector3F &wi) const {
            return 0;
        }

        RENDER_CPU_GPU
        MicrofacetBxDF::MicrofacetBxDF(const Spectrum &Ks, const Spectrum &Kt, Float etaI, Float etaT,
                                       MicrofacetDistribution distribution, const TransportMode mode) :
                _type(BxDFType(BSDF_Glossy | BSDF_Transmission | BSDF_Reflection)), _Ks(Ks), _Kt(Kt),
                _etaI(etaI), _etaT(etaT), _microfacetDistribution(distribution), _mode(mode) {}

        RENDER_CPU_GPU
        Spectrum MicrofacetBxDF::sampleF(const Vector3F &wo, Vector3F *wi, Float *pdf,
                                         BSDFSample &bsdfSample, BxDFType *sampleType) {
            Float cosThetaWo = math::local_coord::vectorCosTheta(wo);
            if (cosThetaWo == 0.0) {
                return Spectrum(0.f);
            }

            // Sample microfacet
            Normal3F wh = _microfacetDistribution.sampleWh(wo, bsdfSample.uv);
            Float cosThetaWoWh = DOT(wo, wh);
            if (cosThetaWoWh <= 0.0f) {
                // Back face
                return Spectrum(0.f);
            }

            // Reflection probability computation
            Float reflectionProb = fresnel::fresnelDielectric(cosThetaWoWh,
                                                              cosThetaWo > 0 ? _etaI : _etaT,
                                                              cosThetaWo > 0 ? _etaT : _etaI);

            Float scatterSample = bsdfSample.u;
            if (scatterSample < reflectionProb) {
                // Reflection
                Vector3F reflectDir = math::reflect(wo, wh);
                if (wo.y * reflectDir.y < 0) {
                    if (pdf != nullptr) {
                        (*pdf) = 0.0;
                    }
                    return Spectrum(0.f);
                }

                if (wi != nullptr) {
                    (*wi) = reflectDir;
                }

                if (pdf != nullptr) {
                    (*pdf) = samplePdf(wo, reflectDir);
                }

                if (sampleType != nullptr) {
                    (*sampleType) = BxDFType(BSDF_Glossy | BSDF_Reflection);
                }
                return f(wo, reflectDir);
            } else {
                // Refraction
                Vector3F refractDir(0);
                Float eta = cosThetaWo > 0 ? _etaI / _etaT : _etaT / _etaI;
                if (!math::refract(wo, wh, eta, &refractDir)) {
                    // Refraction failed
                    return Spectrum(0.);
                }

                refractDir = NORMALIZE(refractDir);
                if (refractDir.y * wo.y > 0) {
                    if (pdf != nullptr) {
                        (*pdf) = 0.f;
                    }
                    return Spectrum(0.0f);
                }

                if (wi != nullptr) {
                    (*wi) = refractDir;
                }

                if (pdf != nullptr) {
                    (*pdf) = samplePdf(wo, refractDir);
                }

                if (sampleType != nullptr) {
                    (*sampleType) = BxDFType(BSDF_Glossy | BSDF_Transmission);
                }
                return f(wo, refractDir);
            }
        }

        RENDER_CPU_GPU
        Spectrum MicrofacetBxDF::f(const Vector3F &wo, const Vector3F &wi) const {
            Float cosThetaWo = math::local_coord::vectorCosTheta(wo);
            Float cosThetaWi = math::local_coord::vectorCosTheta(wi);

            if (cosThetaWo == 0 || cosThetaWi == 0) {
                return Spectrum(0.0f);
            }

            if (cosThetaWo * cosThetaWi < 0) {
                // Refraction
                // Compute half angle
                Float invEta = cosThetaWo > 0 ? _etaT / _etaI : _etaI / _etaT;
                Normal3F wh = NORMALIZE(wo + invEta * wi);
                // Correction for positive hemisphere
                if (wh.y < 0) {
                    wh *= -1;
                }

                if (DOT(wo, wh) * DOT(wi, wh) > 0) {
                    return Spectrum(0.0f);
                }

                Float D_Wh = _microfacetDistribution.D(wh);
                Float G_Wo_Wi = _microfacetDistribution.G(wo, wi, wh);

                Float cosThetaWoWh = DOT(wo, wh);
                Float Fr = fresnel::fresnelDielectric(cosThetaWoWh, _etaI, _etaT);

                Float absCosThetaWiWh = ABS_DOT(wi, wh);
                Float absCosThetaWoWh = ABS_DOT(wo, wh);

                Float cosThetaWiWh = DOT(wi, wh);
                Float term = cosThetaWoWh + invEta * (cosThetaWiWh);

                Float scale = _mode == TransportMode::RADIANCE ? 1.0 / invEta : 1.;
                return _Kt * (1.0f - Fr) *
                       std::abs(((D_Wh * G_Wo_Wi * scale * scale) / (term * term)) *
                                ((absCosThetaWiWh * absCosThetaWoWh) / (cosThetaWo * cosThetaWi)));

            } else {
                // Reflection
                Vector3F wh = NORMALIZE(wo + wi);
                if (wh.y < 0) {
                    wh *= -1;
                }

                Float D_Wh = _microfacetDistribution.D(wh);
                Float G_Wo_Wi = _microfacetDistribution.G(wo, wi, wh);

                Float cosThetaWoWh = DOT(wo, wh);
                Float Fr = fresnel::fresnelDielectric(cosThetaWoWh, _etaI, _etaT);

                return (D_Wh * G_Wo_Wi * Fr * _Ks) / (4 * std::abs(cosThetaWi * cosThetaWo));
            }
        }

        RENDER_CPU_GPU
        Float MicrofacetBxDF::samplePdf(const Vector3F &wo, const Vector3F &wi) const {
            Float cosThetaWo = math::local_coord::vectorCosTheta(wo);
            Float cosThetaWi = math::local_coord::vectorCosTheta(wi);

            if (cosThetaWi == 0 || cosThetaWo == 0) {
                return 0.0f;
            }

            if (cosThetaWi * cosThetaWo > 0) {
                // Reflection
                Vector3F wh = NORMALIZE(wo + wi);
                Float reflectionProb = fresnel::fresnelDielectric(ABS_DOT(wo, wh),
                                                                  cosThetaWo > 0 ? _etaI : _etaT,
                                                                  cosThetaWi > 0 ? _etaT : _etaI);
                return reflectionProb * _microfacetDistribution.samplePdf(wo, wh) / (4 * ABS_DOT(wo, wh));
            } else {
                // Refraction
                Float eta = cosThetaWo > 0 ? _etaI / _etaT : _etaT / _etaI;
                Float invEta = 1.0 / eta;

                Normal3F wh = NORMALIZE(wo + invEta * wi);
                // Transmission check
                if (DOT(wo, wh) * DOT(wi, wh) >= 0) {
                    return 0.f;
                }

                Float refractionProb = 1.0 - fresnel::fresnelDielectric(ABS_DOT(wo, wh),
                                                                        cosThetaWo > 0 ? _etaI : _etaT,
                                                                        cosThetaWi > 0 ? _etaT : _etaI);
                Float sqrtDenom = DOT(wo, wh) + invEta * DOT(wi, wh);
                // Some difference from PBRT
                return refractionProb * _microfacetDistribution.samplePdf(wo, wh) * ABS_DOT(wi, wh) /
                       (sqrtDenom * sqrtDenom);
            }
        }

        RENDER_CPU_GPU
        ConductorBxDF::ConductorBxDF() : _type(BxDFType(BSDF_Reflection | BSDF_Glossy)) {}

        RENDER_CPU_GPU
        ConductorBxDF::ConductorBxDF(const Spectrum &Ks, const Spectrum &etaI, const Spectrum &etaT,
                                     const Spectrum &K, Float alpha, MicrofacetDistribType distribType) :
                _type(BxDFType(BSDF_Reflection | BSDF_Glossy)),
                _Ks(Ks), _K(K), _etaI(etaI), _etaT(etaT) {
            if (distribType == GGX) {
                _distribStorage.set(GGXDistribution(alpha));
                _distribution = (_distribStorage.ptr<GGXDistribution>());
            } else {
                assert(false);
            }
        }

        RENDER_CPU_GPU
        Spectrum ConductorBxDF::f(const Vector3F &wo, const Vector3F &wi) const {
            Float cosThetaO = wo.y;
            Float cosThetaI = wi.y;
            if (cosThetaI == 0 || cosThetaO == 0) {
                return Spectrum(0.0f);
            }

            if (cosThetaI * cosThetaO <= 0) {
                return Spectrum(0.0f);
            }

            // Half angle
            Vector3F wh = NORMALIZE(wo + wi);
            if (wh.y < 0) {
                wh *= -1;
            }

            Float D_Wh = _distribution.D(wh);
            Float G_Wo_Wi = _distribution.G(wo, wi, wh);

            Float cosThetaH = DOT(wi, wh);
            Spectrum Fr = fresnel::fresnelConductor(cosThetaH, _etaI, _etaT, _K);
            return (D_Wh * G_Wo_Wi * Fr * _Ks) / (4 * std::abs(cosThetaO * cosThetaI));
        }

        RENDER_CPU_GPU
        Spectrum ConductorBxDF::sampleF(const Vector3F &wo, Vector3F *wi, Float *pdf,
                                        BSDFSample &bsdfSample, BxDFType *sampleType) {
            if (wo.y == 0.) {
                return Spectrum(0.f);
            }

            Vector3F wh = _distribution.sampleWh(wo, bsdfSample.uv);
            if (DOT(wo, wh) < 0) {
                return Spectrum(0.f);
            }

            // Check same side
            Vector3F reflectDir = math::reflect(wo, wh);

            if ((wo.y * reflectDir.y) <= 0) {
                if (pdf != nullptr) {
                    (*pdf) = 0;
                }
                return Spectrum(0.f);
            }

            if (wi != nullptr) {
                (*wi) = reflectDir;
            }

            if (pdf != nullptr) {
//                    (*pdf) = _microfacetDistribution->samplePdf(wo, wh) / (4 * DOT(wo, wh));
                (*pdf) = samplePdf(wo, reflectDir);
            }

            if (sampleType != nullptr) {
                (*sampleType) = BxDFType(BSDF_Glossy | BSDF_Reflection);
            }
            return f(wo, reflectDir);
        }

        RENDER_CPU_GPU
        Float ConductorBxDF::samplePdf(const Vector3F &wo, const Vector3F &wi) const {
            // Check same hemisphere
            if (wo.y * wi.y <= 0) {
                return 0.;
            }
            Vector3F wh = NORMALIZE(wo + wi);
            return _distribution.samplePdf(wo, wh) / (4 * DOT(wo, wh));
        }

        RENDER_CPU_GPU
        inline Spectrum BxDF::f(const Vector3F &wo, const Vector3F &wi) const {
            auto func = [&](auto ptr) { return ptr->f(wo, wi); };
            return proxyCall(func);
        }

        RENDER_CPU_GPU
        inline Spectrum BxDF::sampleF(const Vector3F &wo, Vector3F *wi, Float *pdf,
                                      BSDFSample &bsdfSample, BxDFType *sampleType) {
            auto func = [&](auto ptr) { return ptr->sampleF(wo, wi, pdf, bsdfSample, sampleType); };
            return proxyCall(func);
        }

        RENDER_CPU_GPU
        inline Float BxDF::samplePdf(const Vector3F &wo, const Vector3F &wi) const {
            auto func = [&](auto ptr) { return ptr->samplePdf(wo, wi); };
            return proxyCall(func);
        }

        RENDER_CPU_GPU
        inline BxDFType BxDF::type() const {
            auto func = [&](auto ptr) { return ptr->type(); };
            return proxyCall(func);
        }

        RENDER_CPU_GPU
        inline bool BxDF::allIncludeOf(const BxDFType bxdfType) const {
            auto func = [&](auto ptr) {
                BxDFType type = ptr->type();
                return (type & bxdfType) == type;
            };
            return proxyCall(func);
        }

        RENDER_CPU_GPU
        inline bool BxDF::hasAllOf(const BxDFType bxdfType) const {
            auto func = [&](auto ptr) {
                BxDFType type = ptr->type();
                return (type & bxdfType) == bxdfType;
            };
            return proxyCall(func);
        }

        RENDER_CPU_GPU
        inline bool BxDF::hasAnyOf(const BxDFType bxdfType) const {
            auto func = [&](auto ptr) {
                BxDFType type = ptr->type();
                return (type & bxdfType) > 0;
            };
            return proxyCall(func);
        }

        RENDER_CPU_GPU
        BSDF::BSDF(const Normal3F &ng, const Normal3F &ns, const Vector3F &wo) {
            // Build tangent space
            _tanY = NORMALIZE(ns);
            _tanZ = NORMALIZE(CROSS(-wo, _tanY));
            if (!math::isValid(_tanZ)) {
                math::tangentSpace(_tanY, &_tanX, &_tanZ);
            } else {
                _tanX = NORMALIZE(CROSS(_tanY, _tanZ));
            }
            _ng = ng;
        }

        RENDER_CPU_GPU
        Vector3F BSDF::toObjectSpace(const Vector3F &v) const {
            return Vector3F(DOT(_tanX, v), DOT(_tanY, v), DOT(_tanZ, v));
        }

        RENDER_CPU_GPU
        Vector3F BSDF::toWorldSpace(const Vector3F &v) const {
            return Vector3F(_tanX.x * v.x + _tanY.x * v.y + _tanZ.x * v.z,
                            _tanX.y * v.x + _tanY.y * v.y + _tanZ.y * v.z,
                            _tanX.z * v.x + _tanY.z * v.y + _tanZ.z * v.z);
        }

        RENDER_CPU_GPU
        void BSDF::setBxDF(BxDF bxdf) {
            _bxdf = bxdf;
        }

        RENDER_CPU_GPU
        Spectrum BSDF::f(const Vector3F &worldWo, const Vector3F &worldWi, BxDFType type) const {
            if (_bxdf.nullable()) {
                return Spectrum(0.f);
            }

            Vector3F wo = toObjectSpace(worldWo);
            Vector3F wi = toObjectSpace(worldWi);

            bool reflect = DOT(worldWo, _ng) * DOT(worldWi, _ng) > 0;
            Spectrum f = Spectrum(0);
            if (_bxdf.allIncludeOf(type)) {
                if ((reflect && _bxdf.hasAllOf(BSDF_Reflection)) ||
                    (!reflect && _bxdf.hasAllOf(BSDF_Transmission))) {
                    f = _bxdf.f(wo, wi);
                }
            }
            return f;
        }

        RENDER_CPU_GPU
        Spectrum BSDF::sampleF(const Vector3F &worldWo, Vector3F *worldWi, Float *pdf,
                               BSDFSample &bsdfSample, BxDFType *sampleType, BxDFType type) {
            if (_bxdf.nullable()) {
                return Spectrum(0.f);
            }

            bool matched = _bxdf.allIncludeOf(type);
            if (!matched) {
                if (sampleType != nullptr) {
                    *sampleType = BxDFType(0);
                }
                return Spectrum(0.f);
            } else {
                if (_bxdf == nullptr) {
                    return Spectrum(0);
                }
                if (sampleType != nullptr) {
                    *sampleType = _bxdf.type();
                }

                Vector3F wo = toObjectSpace(worldWo);
                Vector3F wi = Vector3F(0.0f);
                Float samplePdf = 0.f;

                Spectrum f = _bxdf.sampleF(wo, &wi, &samplePdf, bsdfSample, sampleType);

                if (samplePdf == 0) {
                    if (sampleType != nullptr) {
                        *sampleType = BxDFType(0);
                    }
                    return Spectrum(0.0);
                }

                (*worldWi) = toWorldSpace(wi);
                (*pdf) = samplePdf;
                return f;
            }
        }

        RENDER_CPU_GPU
        Float BSDF::samplePdf(const Vector3F &worldWo, const Vector3F &worldWi, BxDFType type) const {
            if (_bxdf.nullable()) {
                return Float(0);
            }

            Vector3F wo = toObjectSpace(worldWo);
            Vector3F wi = toObjectSpace(worldWi);
            if (std::abs(wo.y - 0) < Epsilon) {
                return 0.0;
            }
            return _bxdf.samplePdf(wo, wi);
        }

        RENDER_CPU_GPU
        int BSDF::allIncludeOf(BxDFType bxdfType) const {
            if (_bxdf.nullable()) {
                return 0;
            }
            return _bxdf.allIncludeOf(bxdfType) ? 1 : 0;
        }

        RENDER_CPU_GPU
        int BSDF::hasAllOf(BxDFType bxdfType) const {
            if (_bxdf.nullable()) {
                return 0;
            }
            return _bxdf.hasAllOf(bxdfType) ? 1 : 0;
        }

        RENDER_CPU_GPU
        int BSDF::hasAnyOf(const BxDFType bxdfType) const {
            if (_bxdf.nullable()) {
                return 0;
            }
            return _bxdf.hasAnyOf(bxdfType) ? 1 : 0;
        }
    }
}