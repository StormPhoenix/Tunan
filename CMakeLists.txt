cmake_minimum_required(VERSION 3.17)
project(Tunan)

set(CMAKE_CXX_STANDARD 17)

INCLUDE(CheckCXXSourceCompiles)
INCLUDE(CheckCXXSourceRuns)

CHECK_CXX_SOURCE_COMPILES("
#include <malloc.h>
int main() {
    void *p = _aligned_malloc(1024, 32);
} " WINDOWS_MALLOC_ALIGNED)
if (WINDOWS_MALLOC_ALIGNED)
    add_definitions(-D RENDER_WINDOWS_MALLOC_ALIGNED)
endif (WINDOWS_MALLOC_ALIGNED)

CHECK_CXX_SOURCE_COMPILES("
#include <stdlib.h>
int main() {
    void *p;
    posix_memalign(&p, 32, 1024);
} " POSIX_MALLOC_ALIGNED)
if (POSIX_MALLOC_ALIGNED)
    add_definitions(-D RENDER_POSIX_MALLOC_ALIGNED)
endif (POSIX_MALLOC_ALIGNED)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/CMake/)
find_package(CUDA)
if (CUDA_FOUND)
    enable_language(CUDA)
    set(CUDA_LINK_LIBRARY ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY})
    set(CUDA_INCLUDE_DIRECTORY ${CUDA_INCLUDE_DIRS})
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_61，code=sm_61;)
    message(STATUS "Found CUDA: ${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}")
    #    add_definitions(-D CUDA_VERSION)
endif (CUDA_FOUND)

find_package(OptiX)
if (OptiX_FOUND)
    set(OPTIX_INCLUDE_DIRECTORY ${OptiX_INCLUDE_DIR})
    message(STATUS "Found OptiX")
endif (OptiX_FOUND)

option(START_CUDA "option for cuda" ON)
#option(START_CUDA "option for cuda" OFF)

set(BUILD_GPU_ENABLED false)
if (CUDA_FOUND AND CUDA_FOUND AND START_CUDA)
    get_filename_component(cuda_compiler_bin ${CMAKE_CUDA_COMPILER} DIRECTORY)
    find_program(BIN2C
            NAMES bin2c
            PATHS ${cuda_compiler_bin}
            NO_DEFAULT_PATH)
    if (BIN2C)
        message(STATUS "Found program bin2c")
    elseif (BIN2C)
        message(FATAL_ERROR "BIN2C not found")
    endif (BIN2C)

    set(BUILD_GPU_ENABLED true)
    add_definitions(-D __BUILD_GPU_RENDER_ENABLE__)
    message(STATUS "Build in GPU Mode.")
endif (CUDA_FOUND AND CUDA_FOUND AND START_CUDA)

set(PROJ_INCLUDE_DIR
        ${PROJECT_SOURCE_DIR}/include)

add_library (cuda_build_configuration INTERFACE)
if (BUILD_GPU_ENABLED)
    target_compile_options (
            cuda_build_configuration
            INTERFACE
            "$<$<COMPILE_LANGUAGE:CUDA>:--std=c++17;--use_fast_math;--expt-relaxed-constexpr;--extended-lambda;--forward-unknown-to-host-compiler>"
    )
#    --std=c++17;
endif(BUILD_GPU_ENABLED)

### Compile *.PTX code
if (BUILD_GPU_ENABLED)
    set(PTX_CODE_GEN ptx_code_gen)
    set(OPTIX_RAYTRACING_FILE
            src/tunan/gpu/optix_ray.cu)
    add_library(${PTX_CODE_GEN} OBJECT ${OPTIX_RAYTRACING_FILE})
    set_property(TARGET ${PTX_CODE_GEN} PROPERTY CUDA_PTX_COMPILATION ON)
    target_compile_options("${PTX_CODE_GEN}"
            PRIVATE
            -Xcudafe=--display_error_number -Xcudafe=--diag_suppress=3089)
    target_include_directories(${PTX_CODE_GEN}
            PRIVATE
            src
            ${CUDA_INCLUDE_DIRECTORY}
            ${OPTIX_INCLUDE_DIRECTORY}
            ${PROJ_INCLUDE_DIR})
    target_link_libraries(${PTX_CODE_GEN}
            PRIVATE
            ${CUDA_LINK_LIBRARY})

    set(c_ptx_code_val_name "OptixPtxCode")
    set(OPTIX_EMBEDDED_FILE ${OPTIX_RAYTRACING_FILE}.ptx_embedded.cpp)
    add_custom_command(
            OUTPUT "${OPTIX_EMBEDDED_FILE}"
            COMMAND ${CMAKE_COMMAND}
            "-DBIN_TO_C_COMMAND=${BIN2C}"
            "-DOBJECTS=$<TARGET_OBJECTS:${PTX_CODE_GEN}>"
            "-DVAR_NAME=${c_ptx_code_val_name}"
            "-DOUTPUT=${OPTIX_EMBEDDED_FILE}"
            -P ${CMAKE_MODULE_PATH}/bin2c_wrapper.cmake
            VERBATIM
            DEPENDS "${PTX_CODE_GEN}" $<TARGET_OBJECTS:${PTX_CODE_GEN}>
    )

    message(STATUS "Build *.PTX embedded code: ${OPTIX_EMBEDDED_FILE}")
    add_library(render_ptx_lib STATIC ${OPTIX_EMBEDDED_FILE})
endif (BUILD_GPU_ENABLED)

set(PUGIXML_SRCS
        include/ext/pugixml/pugixml.cpp)

set(TuNan_SRCS
        src/tunan/parallel/parallels.cpp
        src/tunan/tracer/path.cpp
        src/tunan/sampler/SamplerFactory.cpp
        src/tunan/sampler/samplers.cpp
        src/tunan/material/materials.cpp
        src/tunan/material/bsdfs.cpp
        src/tunan/material/textures.cpp
        src/tunan/material/mappings.cpp
        src/tunan/material/fresnels.cpp
        src/tunan/material/microfacets.cpp
        src/tunan/scene/Ray.cpp
        src/tunan/scene/film.cpp
        src/tunan/scene/shapes.cpp
        src/tunan/scene/lights.cpp
        src/tunan/scene/cameras.cpp
        src/tunan/scene/importers.cpp
        src/tunan/scene/scenedata.cpp
        src/tunan/scene/meshes.cpp)

set(TuNan_HDRS
        include/tunan/common.h
        include/tunan/sampler/rng.h
        include/tunan/material/materials.h
        include/tunan/medium/Medium.h)

set(TuNan_BASE_SRCS
        src/tunan/base/distributions.cpp
        src/tunan/base/interactions.cpp
        src/tunan/base/transform.cpp)
set(TuNan_BASE_HDRS
        include/tunan/base/spectrum.h
        include/tunan/base/containers.h
        include/tunan/base/mediums.h)

set(TuNan_UTIL_SRCS
        src/tunan/utils/model_loader.cpp
        src/tunan/utils/image_utils.cpp
        src/tunan/utils/file_utils.cpp
        src/tunan/utils/ResourceManager.cpp
        src/tunan/utils/memory/HostAllocator.cpp)

set(TuNan_UTIL_HDRS
        include/tunan/utils/type_utils.h
        include/tunan/utils/TaggedPointer.h)


### Set cuda properties
if (BUILD_GPU_ENABLED)
    set(TuNan_UTIL_SRCS
            ${TuNan_UTIL_SRCS}
            src/tunan/utils/memory/CudaAllocator.cpp)

    set(TuNan_SRCS
            ${TuNan_SRCS}
            src/tunan/scene/OptixIntersectable.cpp)

    set_source_files_properties(
            include/tunan/utils/TaggedPointer.h
            include/tunan/math.h
            src/tunan/utils/model_loader.cpp
            main.cpp
            ${TuNan_SRCS}
            ${TuNan_HDRS}
            ${TuNan_BASE_SRCS}
            ${TuNan_BASE_HDRS}
            ${TuNan_UTIL_SRCS}
            ${TuNan_UTIL_HDRS}
            PROPERTIES LANGUAGE CUDA)
endif ()

set(TuNan_LIB tunan_lib)
add_library(${TuNan_LIB} STATIC
        ${TuNan_SRCS}
        ${TuNan_HDRS}
        ${TuNan_BASE_SRCS}
        ${TuNan_BASE_HDRS}
        ${TuNan_UTIL_SRCS}
        ${TuNan_UTIL_HDRS})
target_include_directories(${TuNan_LIB}
        PRIVATE
        ${PROJ_INCLUDE_DIR})
set_property(TARGET ${TuNan_LIB} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries (${TuNan_LIB} PRIVATE $<$<BOOL:PBRT_CUDA_ENABLED>:cuda_build_configuration>)

if (BUILD_GPU_ENABLED)
    set_property (TARGET ${TuNan_LIB} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    add_dependencies(${TuNan_LIB} render_ptx_lib)
    target_link_libraries(${TuNan_LIB}
            PRIVATE
            render_ptx_lib
            ${CUDA_LINK_LIBRARY})

    target_include_directories(${TuNan_LIB}
            PRIVATE
            ${OPTIX_INCLUDE_DIRECTORY}
            ${CUDA_INCLUDE_DIRECTORY})
endif (BUILD_GPU_ENABLED)
set_property (TARGET ${TuNan_LIB} PROPERTY CUDA_SEPARABLE_COMPILATION ON)

set(PUGIXML_LIB pugixml)
add_library(${PUGIXML_LIB} ${PUGIXML_SRCS})
target_include_directories(${PUGIXML_LIB}
        PRIVATE
        include/ext)

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(
        ${PROJECT_NAME}
        PRIVATE
        ${PUGIXML_LIB}
        ${TuNan_LIB})
target_include_directories(
        ${PROJECT_NAME}
        PRIVATE
        ${OPTIX_INCLUDE_DIRECTORY}
        ${CUDA_INCLUDE_DIRECTORY}
        ${PROJ_INCLUDE_DIR})