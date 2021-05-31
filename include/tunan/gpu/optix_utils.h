//
// Created by StormPhoenix on 2021/5/31.
//

#ifndef TUNAN_OPTIX_UTILS_H
#define TUNAN_OPTIX_UTILS_H

#include <optix.h>

#define OPTIX_CHECK( call )                                                    \
    do                                                                         \
    {                                                                          \
        OptixResult res = call;                                                \
        if( res != OPTIX_SUCCESS )                                             \
        {                                                                      \
            ASSERT(false, "Optix call error: " + std::string(optixGetErrorString(res)));  \
        }                                                                      \
    } while( 0 )


#define OPTIX_CHECK_LOG(call, log)                                             \
    do {                                                                            \
        OptixResult res = call;                                                     \
        if (res != OPTIX_SUCCESS) {                                            \
            std::stringstream ss;                                              \
            ss << "Optix call error: error code: " << res << " msg: "          \
                << (optixGetErrorString(res)) << " " << log << std::endl;                     \
            ASSERT(false, ss.str());                                                                  \
        }   \
    } while (false)


#endif //TUNAN_OPTIX_UTILS_H
