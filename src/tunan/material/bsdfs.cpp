//
// Created by StormPhoenix on 2021/6/7.
//

#include <tunan/math.h>
#include <tunan/material/bsdfs.h>
#include <tunan/sampler/samplers.h>

namespace RENDER_NAMESPACE {
    namespace bsdf {

        RENDER_CPU_GPU
        LambertianBxDF::LambertianBxDF(const Spectrum &Kd) :
                _type(BxDFType(BSDF_DIFFUSE | BSDF_REFLECTION)), _Kd(Kd) {}

        RENDER_CPU_GPU
        Spectrum LambertianBxDF::f(const Vector3F &wo, const Vector3F &wi) const {
            return _Kd * Inv_Pi;
        }

        RENDER_CPU_GPU
        Spectrum LambertianBxDF::sampleF(const Vector3F &wo, Vector3F *wi, Float *pdf,
                                         Vector2F uv, BxDFType *sampleType) {
            *wi = sampler::hemiCosineSampling(uv);
            if (wo.y < 0) {
                wi->y *= -1;
            }
            *pdf = samplePdf(wo, *wi);
            if (sampleType != nullptr) {
                (*sampleType) = BxDFType(BSDF_DIFFUSE | BSDF_REFLECTION);
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
        BxDFType LambertianBxDF::type() const {
            return _type;
        }

        RENDER_CPU_GPU
        inline Spectrum BxDF::f(const Vector3F &wo, const Vector3F &wi) const {
            auto func = [&](auto ptr) { return ptr->f(wo, wi); };
            return proxyCall(func);
        }

        RENDER_CPU_GPU
        inline Spectrum BxDF::sampleF(const Vector3F &wo, Vector3F *wi, Float *pdf,
                                      Vector2F uv, BxDFType *sampleType) {
            auto func = [&](auto ptr) { return ptr->sampleF(wo, wi, pdf, uv, sampleType); };
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
            Vector3F wo = toObjectSpace(worldWo);
            Vector3F wi = toObjectSpace(worldWi);

            bool reflect = DOT(worldWo, _ng) * DOT(worldWi, _ng) > 0;
            Spectrum f = Spectrum(0);
            if (_bxdf.allIncludeOf(type)) {
                if ((reflect && _bxdf.hasAllOf(BSDF_REFLECTION)) ||
                    (!reflect && _bxdf.hasAllOf(BSDF_TRANSMISSION))) {
                    f = _bxdf.f(wo, wi);
                }
            }
            return f;
        }

        RENDER_CPU_GPU
        Spectrum BSDF::sampleF(const Vector3F &worldWo, Vector3F *worldWi, Float *pdf,
                               Vector2F uv, BxDFType *sampleType, BxDFType type)  {
            bool matched = _bxdf.allIncludeOf(type);
            if (!matched) {
                // 没有类型被匹配上
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
                Spectrum f = _bxdf.sampleF(wo, &wi, &samplePdf, uv, sampleType);

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
            if (_bxdf == nullptr) {
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
            if (_bxdf == nullptr) {
                return 0;
            }
            return _bxdf.allIncludeOf(bxdfType) ? 1 : 0;
        }

        RENDER_CPU_GPU
        int BSDF::hasAllOf(BxDFType bxdfType) const {
            if (_bxdf == nullptr) {
                return 0;
            }
            return _bxdf.hasAllOf(bxdfType) ? 1 : 0;
        }

        RENDER_CPU_GPU
        int BSDF::hasAnyOf(const BxDFType bxdfType) const {
            if (_bxdf == nullptr) {
                return 0;
            }
            return _bxdf.hasAnyOf(bxdfType) ? 1 : 0;
        }
    }
}