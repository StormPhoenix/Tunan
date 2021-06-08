//
// Created by StormPhoenix on 2021/6/7.
//

#ifndef TUNAN_BSDFS_H
#define TUNAN_BSDFS_H

#include <tunan/common.h>
#include <tunan/base/spectrum.h>
#include <tunan/utils/TaggedPointer.h>

namespace RENDER_NAMESPACE {
    namespace bsdf {
        using base::Spectrum;
        using utils::TaggedPointer;

        typedef enum TransportMode {
            IMPORTANCE,
            RADIANCE
        } TransportMode;

        typedef enum BxDFType {
            BSDF_REFLECTION = 1 << 0,
            BSDF_DIFFUSE = 1 << 1,
            BSDF_GLOSSY = 1 << 2,
            BSDF_SPECULAR = 1 << 3,
            BSDF_TRANSMISSION = 1 << 4,
            BSDF_ALL = BSDF_REFLECTION | BSDF_DIFFUSE | BSDF_GLOSSY | BSDF_SPECULAR | BSDF_TRANSMISSION
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
                                    Vector2F uv, BxDFType *sampleType);

            RENDER_CPU_GPU
            inline Float samplePdf(const Vector3F &wo, const Vector3F &wi) const;

            RENDER_CPU_GPU
            inline BxDFType type() const;

            RENDER_CPU_GPU
            inline ~LambertianBxDF() {}

        private:
            Spectrum _Kd;
            BxDFType _type;
        };

        class BxDF : public TaggedPointer<LambertianBxDF> {
        public:
            using TaggedPointer::TaggedPointer;

            RENDER_CPU_GPU
            inline Spectrum f(const Vector3F &wo, const Vector3F &wi) const;

            RENDER_CPU_GPU
            inline Spectrum sampleF(const Vector3F &wo, Vector3F *wi, Float *pdf,
                                    Vector2F uv, BxDFType *sampleType);

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
            Spectrum f(const Vector3F &worldWo, const Vector3F &worldWi, BxDFType type = BSDF_ALL) const;

            RENDER_CPU_GPU
            Spectrum sampleF(const Vector3F &worldWo, Vector3F *worldWi, Float *pdf,
                             Vector2F uv, BxDFType *sampleType = nullptr,
                             BxDFType type = BSDF_ALL);

            RENDER_CPU_GPU
            Float samplePdf(const Vector3F &worldWo, const Vector3F &worldWi, BxDFType type = BSDF_ALL) const;

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
