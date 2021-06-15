//
// Created by StormPhoenix on 2021/5/31.
//

#ifndef TUNAN_TYPE_UTILS_H
#define TUNAN_TYPE_UTILS_H

#include <tunan/common.h>
#include <type_traits>

template<typename... Ts>
struct TypeList {
    constexpr static int count = sizeof...(Ts);
};

template<bool B, typename T, typename R>
struct IfThenElse;

template<typename T, typename R>
struct IfThenElse<true, T, R> {
    using Type = T;
};

template<typename T, typename R>
struct IfThenElse<false, T, R> {
    using Type = R;
};

template<typename List>
struct IsEmpty {
    constexpr static bool value = false;
};

template<>
struct IsEmpty<TypeList<>> {
    constexpr static bool value = true;
};

template<typename T>
struct GetFirst;

template<typename Head, typename... Tails>
struct GetFirst<TypeList<Head, Tails...>> {
    using Type = Head;
};

template<typename List>
using GetFirstT = typename GetFirst<List>::Type;

template<typename List>
struct PopFirst;

template<typename Head, typename... Tails>
struct PopFirst<TypeList<Head, Tails...>> {
    using Type = TypeList<Tails...>;
};

template<typename List>
using PopFirstT = typename PopFirst<List>::Type;

template<typename List, unsigned N>
struct NthType : public NthType<PopFirstT<List>, N - 1> {
};

template<typename List>
struct NthType<List, 0> : public GetFirst<List> {
};

template<typename List, unsigned N>
using NthTypeT = typename NthType<List, N>::Type;

template<typename List>
struct LargestType {
private:
    using First = GetFirstT<List>;
    using Rest = typename LargestType<PopFirstT<List>>::Type;
public:
    using Type = typename IfThenElse<(sizeof(First) >= sizeof(Rest)), First, Rest>::Type;
};

template<>
struct LargestType<TypeList<>> {
    using Type = char;
};

template<typename Ts>
using LargestTypeT = typename LargestType<Ts>::Type;

template<typename List, typename T, unsigned N = 0, bool Empty = IsEmpty<List>::value>
struct FindIndexOf;

template<typename List, typename T, unsigned N>
struct FindIndexOf<List, T, N, false> :
        public IfThenElse<std::is_same<GetFirstT<List>, T>::value,
                std::integral_constant<unsigned, N>,
                FindIndexOf<PopFirstT<List>, T, N + 1>>::Type {
};

template<typename List, typename T, unsigned N>
struct FindIndexOf<List, T, N, true> {
};

template<typename List, typename T>
struct HasType;

template<typename List, typename T>
struct HasType {
    constexpr static bool value = std::is_same<GetFirstT<List>, T>::value ? true : HasType<PopFirstT<List>, T>::value;
};

template<typename T>
struct HasType<TypeList<>, T> {
    constexpr static bool value = false;
};

template<typename... Ts>
class VariantStorage {
private:
    using LargestT = LargestTypeT<TypeList<Ts...>>;
    alignas(Ts...) unsigned char buffer[sizeof(sizeof(LargestT))];
    unsigned discriminator = 0;
public:
    unsigned getDiscriminator() const {
        return discriminator;
    }

    void setDiscriminator(unsigned val) {
        discriminator = val;
    }

    template<typename T>
    T *getBufferAs() {
        return reinterpret_cast<T *>(buffer);
    }

    template<typename T>
    T const *getBufferAs() const {
        return reinterpret_cast<T const *>(buffer);
    }

    void *getRawBuffer() {
        return buffer;
    }

    const void *getRayBuffer() const {
        return buffer;
    }
};

template<typename... Ts>
class Variant;

template<typename T, typename... Ts>
class VariantChoice {
protected:
    using Derived = Variant<Ts...>;

    Derived &getDerived() {
        return *static_cast<Derived *>(this);
    }

    Derived const &getDerived() const {
        return *static_cast<Derived const *>(this);
    }

    constexpr static unsigned discriminator = FindIndexOf<TypeList<Ts...>, T>::value + 1;
public:
    VariantChoice() {}

    VariantChoice(T const &value) {
        new(getDerived().getRawBuffer()) T(value);
        getDerived().setDiscriminator(discriminator);
    }

    VariantChoice(T &&value) {
        new(getDerived().getRawBuffer()) T(std::move(value));
        getDerived().setDiscriminator(discriminator);
    }

    bool destroy() {
        if (getDerived().getDiscriminator() == discriminator) {
            getDerived().template getBufferAs<T>()->~T();
            return true;
        }
        return false;
    }
};

class EmptyVariant : public std::exception {
};

template<typename... Ts>
class Variant : private VariantStorage<Ts...>, private VariantChoice<Ts, Ts...> ... {
    template<typename T, typename... OtherTs>
    friend
    class VariantChoice;

public:
    template<typename T>
    bool is() const {
        return this->getDiscriminator() == VariantChoice<T, Ts...>::discriminator;
    }

    template<typename T>
    void set(T const &value) {
        static_assert((HasType<TypeList<Ts...>, T>::value), "T not in type list.");
        if (this->getDiscriminator() == (FindIndexOf<TypeList<Ts...>, T>::value + 1)) {
            (*(this->getBufferAs<T>())) = value;
        } else {
            destroy();
            new(this->getRawBuffer()) T(value);
            this->setDiscriminator((FindIndexOf<TypeList<Ts...>, T>::value + 1));
        }
    }

    template<typename T>
    void set(T &&value) {
        static_assert((HasType<TypeList<Ts...>, T>::value), "T not in type list.");
        if (this->getDiscriminator() == (FindIndexOf<TypeList<Ts...>, T>::value + 1)) {
            (*(this->getBufferAs<T>())) = std::move(value);
        } else {
            destroy();
            new(this->getRawBuffer()) T(std::move(value));
            this->setDiscriminator((FindIndexOf<TypeList<Ts...>, T>::value + 1));
        }
    }

    template<typename T>
    T *ptr() {
        if (empty()) {
            throw EmptyVariant();
        }
        assert(is<T>());
        return this->template getBufferAs<T>();
    }

    template<typename T>
    T &get() {
        if (empty()) {
            throw EmptyVariant();
        }
        assert(is<T>());
        return *this->template getBufferAs<T>();
    }

    template<typename T>
    T const &get() const {
        if (empty()) {
            throw EmptyVariant();
        }
        assert(is<T>());
        return *this->template getBufferAs<T>();
    }

    bool empty() const {
        return this->getDiscriminator() == 0;
    }

    Variant() {}

    template<typename T>
    Variant(T const &value) {
        new(getRawBuffer()) T(value);
        setDiscriminator((FindIndexOf<TypeList<Ts...>, T>::value + 1));
    }

    template<typename T>
    Variant(T &&value) {
        new(getRawBuffer()) T(std::move(value));
        setDiscriminator((FindIndexOf<TypeList<Ts...>, T>::value + 1));
    }

    ~Variant() { destroy(); }

    void destroy() {
        bool results[] = {VariantChoice<Ts, Ts...>::destroy()...};
        this->setDiscriminator(0);
    }
};


template<int n>
struct EvaluateTpType;

template<>
struct EvaluateTpType<1> {
    template<typename F, typename Tp, typename... Ts>
    RENDER_CPU_GPU
    inline auto operator()(F func, Tp tp, int index, TypeList<Ts...> types) {
        static_assert(sizeof...(Ts) >= 1, "Types can be zero. ");
        using T = GetFirstT<TypeList<Ts...>>;
        return func(tp.template cast<T>());
    }
};

template<int n>
struct EvaluateTpType {
    template<typename F, typename TP, typename... Ts>
    RENDER_CPU_GPU
    inline auto operator()(F func, TP tp, int index, TypeList<Ts...> types) {
        if (index > 1) {
            using RestType = PopFirstT<TypeList<Ts...>>;
            return EvaluateTpType<n - 1>()(func, tp, index - 1, RestType());
        } else {
            return EvaluateTpType<1>()(func, tp, index, types);
        }
    }
};

template<typename FWrapper, typename... Ts>
void forEachType(FWrapper wrapper, TypeList<Ts...>);

template<typename FWrapper, typename T, typename... Ts>
void forEachType(FWrapper wrapper, TypeList<T, Ts...>) {
    wrapper.template operator()<T>();
    forEachType(wrapper, TypeList<Ts...>());
}

template<typename FWrapper>
void forEachType(FWrapper wrapper, TypeList<>) {}

#endif //TUNAN_TYPE_UTILS_H
