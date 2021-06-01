//
// Created by Graphics on 2021/5/31.
//

#ifndef TUNAN_COMMON_H
#define TUNAN_COMMON_H

#ifdef __RENDER_GPU_MODE__
#define GLM_FORCE_CUDA
#define RENDER_CPU_GPU __host__ __device__
#define RENDER_GPU __device__
#else
#define RENDER_CPU_GPU
#define RENDER_GPU
#endif

#if defined(_RENDER_DATA_DOUBLE_)
using Float = double;
#else
using Float = float;
#endif

#define CHECK(exp) assert(exp)

#define ASSERT(condition, description) \
    do { \
        if (!(condition)) {                \
            std::cerr << std::endl << "Assertion occured at " << std::endl \
                << __FUNCTION__ << " " << __FILE__ << " " << __LINE__ << "(" << description << ")", \
                throw std::runtime_error(description); \
        } \
    } while (0);

#define RENDER_NAMESPACE tunan

#endif //TUNAN_COMMON_H
