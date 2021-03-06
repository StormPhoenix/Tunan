//
// Created by StormPhoenix on 2021/5/31.
//

#ifndef TUNAN_TAGGEDPOINTER_H
#define TUNAN_TAGGEDPOINTER_H

#include <tunan/common.h>
#include <tunan/utils/type_utils.h>
#include <iostream>

namespace RENDER_NAMESPACE {
    namespace utils {
        template<typename... Ts>
        class TaggedPointer {
        public:
            using Types = TypeList<Ts...>;

            RENDER_CPU_GPU
            TaggedPointer() = default;

            template<typename T>
            RENDER_CPU_GPU TaggedPointer(T *ptr) {
                uintptr_t p = reinterpret_cast<uintptr_t>(ptr);
//                ASSERT((p & ptrMask) == p, "TaggedPointer address too large. ");
                assert((p & ptrMask) == p);
                constexpr unsigned int typeId = typeIndex<T>();
                taggedPtr = p | ((uintptr_t) typeId << typeShift);
            }

            RENDER_CPU_GPU TaggedPointer(decltype(__nullptr) p) {}

            RENDER_CPU_GPU TaggedPointer(const TaggedPointer &tp) {
                taggedPtr = tp.taggedPtr;
            }

            RENDER_CPU_GPU TaggedPointer(TaggedPointer &tp) {
                taggedPtr = tp.taggedPtr;
            }

            RENDER_CPU_GPU
            TaggedPointer &operator=(const TaggedPointer &tp) {
                taggedPtr = tp.taggedPtr;
                return *this;
            }

            RENDER_CPU_GPU
            bool operator==(const TaggedPointer &tp) const {
                return taggedPtr == tp.taggedPtr;
            }

            RENDER_CPU_GPU
            bool operator!=(const TaggedPointer &tp) const {
                return taggedPtr != tp.taggedPtr;
            }

            template<typename T>
            RENDER_CPU_GPU
            static constexpr unsigned int typeIndex() {
                using Tp = typename std::remove_cv_t<T>;
                if (std::is_same<Tp, std::nullptr_t>::value) {
                    // nullptr
                    return 0;
                } else {
                    return 1 + FindIndexOf<Types, Tp>::value;
                }
            }

            RENDER_CPU_GPU unsigned int typeId() const {
                return ((taggedPtr & typeMask) >> typeShift);
            }

            RENDER_CPU_GPU void *ptr() const {
                return reinterpret_cast<void *>(taggedPtr & ptrMask);
            }

            template<typename T>
            RENDER_CPU_GPU bool isType() const {
                return typeId() == typeIndex<T>();
            }

            template<typename T>
            RENDER_CPU_GPU T *cast() {
                assert(isType<T>());
//                ASSERT(isType<T>(), "TaggedPointer type can not be cast. ");
                return reinterpret_cast<T *>(ptr());
            }

            template<typename T>
            RENDER_CPU_GPU const T *cast() const {
                assert(isType<T>());
//                ASSERT(isType<T>(), "TaggedPointer type can not be cast. ");
                return reinterpret_cast<T *>(ptr());
            }

            RENDER_CPU_GPU bool nullable() const {
                return taggedPtr == 0;
            }

            template<typename F>
            RENDER_CPU_GPU inline auto proxyCall(F func) {
                return EvaluateTpType<maxTag()>()(func, *this, typeId(), Types());
            }

            template<typename F>
            RENDER_CPU_GPU inline auto proxyCall(F func) const {
                return EvaluateTpType<maxTag()>()(func, *this, typeId(), Types());
            }

            RENDER_CPU_GPU static constexpr unsigned int maxTag() {
                return sizeof...(Ts);
            }

        private:
            static constexpr int typeBits = 7;
            static constexpr int typeShift = 64 - typeBits;
            static constexpr uint64_t typeMask = ((1ull << typeBits) - 1) << typeShift;
            static constexpr uint64_t ptrMask = ~typeMask;
            uintptr_t taggedPtr = 0;
        };
    }
}

#endif //TUNAN_TAGGEDPOINTER_H
