//
// Created by StormPhoenix on 2021/6/7.
//

#ifndef TUNAN_BSDFS_H
#define TUNAN_BSDFS_H

#include <tunan/common.h>
#include <tunan/base/spectrum.h>
#include <tunan/material/microfacets.h>
#include <tunan/utils/TaggedPointer.h>

namespace RENDER_NAMESPACE {
    namespace bsdf {
        using base::Spectrum;
        using utils::TaggedPointer;
        using namespace microfacet;

        typedef struct BSDFSample {
            Float u;
            Point2F uv;
        } BSDFSample;

        typedef enum TransportMode {
            IMPORTANCE,
            RADIANCE
        } TransportMode;

        typedef enum BxDFType {
            BSDF_Reflection = 1 << 0,
            BSDF_Diffuse = 1 << 1,
            BSDF_Glossy = 1 << 2,
            BSDF_Specular = 1 << 3,
            BSDF_Transmission = 1 << 4,
            BSDF_All = BSDF_Reflection | BSDF_Diffuse | BSDF_Glossy | BSDF_Specular | BSDF_Transmission
        } BxDFType;

        class LambertianBxDF {
        public:
            RENDER_CPU_GPU
            LambertianBxDF();

            RENDER_CPU_GPU
            LambertianBxDF(const Spectrum &Kd);

            RENDER_CPU_GPU
            inline Spectrum f(const Vector3F &wo, const Vector3F &wi) const;

            RENDER_CPU_GPU
            inline Spectrum sampleF(const Vector3F &wo, Vector3F *wi, Float *pdf,
                                    BSDFSample &bsdfSample, BxDFType *sampleType);

            RENDER_CPU_GPU
            inline Float samplePdf(const Vector3F &wo, const Vector3F &wi) const;

            RENDER_CPU_GPU
            inline BxDFType type() const {
                return _type;
            }

            RENDER_CPU_GPU
            inline ~LambertianBxDF() {}

        private:
            Spectrum _Kd;
            BxDFType _type;
        };

        class FresnelSpecularBxDF {
        public:
            RENDER_CPU_GPU
            FresnelSpecularBxDF();

            RENDER_CPU_GPU
            FresnelSpecularBxDF(const Spectrum &reflectance,
                                const Spectrum &transmittance,
                                Float thetaI, Float thetaT,
                                TransportMode mode = TransportMode::RADIANCE);

            RENDER_CPU_GPU
            Spectrum f(const Vector3F &wo, const Vector3F &wi) const;

            RENDER_CPU_GPU
            inline Spectrum sampleF(const Vector3F &wo, Vector3F *wi, Float *pdf,
                                    BSDFSample &bsdfSample, BxDFType *sampleType);

            RENDER_CPU_GPU
            inline Float samplePdf(const Vector3F &wo, const Vector3F &wi) const;

            RENDER_CPU_GPU
            inline BxDFType type() const {
                return _type;
            }

            RENDER_CPU_GPU
            inline ~FresnelSpecularBxDF() {}

        private:
            Float _thetaI;
            Float _thetaT;
            Spectrum _reflectance;
            Spectrum _transmittance;
            TransportMode _mode;
            BxDFType _type;
        };

        class SpecularReflectionBxDF {
        public:
            RENDER_CPU_GPU
            SpecularReflectionBxDF();

            RENDER_CPU_GPU
            SpecularReflectionBxDF(const Spectrum &Ks);

            RENDER_CPU_GPU
            Spectrum f(const Vector3F &wo, const Vector3F &wi) const;

            RENDER_CPU_GPU
            Spectrum sampleF(const Vector3F &wo, Vector3F *wi, Float *pdf,
                             BSDFSample &bsdfSample, BxDFType *sampleType);

            RENDER_CPU_GPU
            Float samplePdf(const Vector3F &wo, const Vector3F &wi) const;

            RENDER_CPU_GPU
            inline BxDFType type() const {
                return _type;
            }

            RENDER_CPU_GPU
            inline ~SpecularReflectionBxDF() {}

        private:
            Spectrum _Ks;
            BxDFType _type;
        };

        class MicrofacetBxDF {
        public:
            RENDER_CPU_GPU
            MicrofacetBxDF(const Spectrum &Ks, const Spectrum &Kt,
                           Float etaI, Float etaT, MicrofacetDistribution distribution,
                           const TransportMode mode);

            RENDER_CPU_GPU
            Spectrum sampleF(const Vector3F &wo, Vector3F *wi, Float *pdf,
                             BSDFSample &bsdfSample, BxDFType *sampleType);

            RENDER_CPU_GPU
            Spectrum f(const Vector3F &wo, const Vector3F &wi) const;

            RENDER_CPU_GPU
            Float samplePdf(const Vector3F &wo, const Vector3F &wi) const;

            RENDER_CPU_GPU
            inline BxDFType type() const {
                return _type;
            }

        private:
            BxDFType _type;
            Float _etaT, _etaI;
            const Spectrum _Ks, _Kt;
            const TransportMode _mode;
            const MicrofacetDistribution _microfacetDistribution;
        };

        class ConductorBxDF {
        public:
            RENDER_CPU_GPU
            ConductorBxDF();

            RENDER_CPU_GPU
            ConductorBxDF(const Spectrum &Ks, const Spectrum &etaI, const Spectrum &etaT,
                          const Spectrum &K, Float alpha, MicrofacetDistribType distribType = GGX);

            RENDER_CPU_GPU
            Spectrum f(const Vector3F &wo, const Vector3F &wi) const;

            RENDER_CPU_GPU
            Spectrum sampleF(const Vector3F &wo, Vector3F *wi, Float *pdf,
                             BSDFSample &bsdfSample, BxDFType *sampleType);

            RENDER_CPU_GPU
            Float samplePdf(const Vector3F &wo, const Vector3F &wi) const;

            RENDER_CPU_GPU
            inline BxDFType type() const {
                return _type;
            }

        private:
            using __MicrofacetDistribType__ = Variant<GGXDistribution>;
            Spectrum _Ks, _K;
            Spectrum _etaI, _etaT;
            __MicrofacetDistribType__ _distribStorage;
            MicrofacetDistribution _distribution;
            BxDFType _type;
        };

        class BxDF : public TaggedPointer<LambertianBxDF, FresnelSpecularBxDF, SpecularReflectionBxDF,
                MicrofacetBxDF, ConductorBxDF> {
        public:
            using TaggedPointer::TaggedPointer;

            RENDER_CPU_GPU
            inline Spectrum f(const Vector3F &wo, const Vector3F &wi) const;

            RENDER_CPU_GPU
            inline Spectrum sampleF(const Vector3F &wo, Vector3F *wi, Float *pdf,
                                    BSDFSample &bsdfSample, BxDFType *sampleType);

            RENDER_CPU_GPU
            inline Float samplePdf(const Vector3F &wo, const Vector3F &wi) const;

            RENDER_CPU_GPU
            inline BxDFType type() const;

            RENDER_CPU_GPU
            inline ~BxDF() {}

            RENDER_CPU_GPU
            inline bool allIncludeOf(const BxDFType bxdfType) const;

            RENDER_CPU_GPU
            inline bool hasAllOf(const BxDFType bxdfType) const;

            RENDER_CPU_GPU
            inline bool hasAnyOf(const BxDFType bxdfType) const;
        };

        class BSDF {
        public:
            RENDER_CPU_GPU
            BSDF(const Normal3F &ng, const Normal3F &ns, const Vector3F &wo);

            RENDER_CPU_GPU
            void setBxDF(BxDF bxdf);

            RENDER_CPU_GPU
            Vector3F toObjectSpace(const Vector3F &v) const;

            RENDER_CPU_GPU
            Vector3F toWorldSpace(const Vector3F &v) const;

            RENDER_CPU_GPU
            Spectrum f(const Vector3F &worldWo, const Vector3F &worldWi, BxDFType type = BSDF_All) const;

            RENDER_CPU_GPU
            Spectrum sampleF(const Vector3F &worldWo, Vector3F *worldWi, Float *pdf,
                             BSDFSample &bsdfSample, BxDFType *sampleType = nullptr,
                             BxDFType type = BSDF_All);

            RENDER_CPU_GPU
            Float samplePdf(const Vector3F &worldWo, const Vector3F &worldWi, BxDFType type = BSDF_All) const;

            RENDER_CPU_GPU
            int allIncludeOf(BxDFType bxdfType) const;

            RENDER_CPU_GPU
            int hasAllOf(BxDFType bxdfType) const;

            RENDER_CPU_GPU
            int hasAnyOf(const BxDFType bxdfType) const;

        private:
            BxDF _bxdf;
            Vector3F _tanY;
            Vector3F _tanX;
            Vector3F _tanZ;
            Vector3F _ng;
        };
    }
}

#endif //TUNAN_BSDFS_H
