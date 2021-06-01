cmake_minimum_required(VERSION 3.13)
project(Tunan)

set(CMAKE_CXX_STANDARD 14)

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
    set(CUDA_LINK_LIBRARY ${CUDA_LIBRARIES})
    set(CUDA_INCLUDE_DIRECTORY ${CUDA_INCLUDE_DIRS})
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_61，code=sm_61;)
    message(STATUS "Found CUDA: ${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}")
    #    add_definitions(-D CUDA_VERSION)
endif (CUDA_FOUND)

## TODO delete
message(STATUS ${CUDA_LINK_LIBRARY})

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
    add_definitions(-D __RENDER_GPU_MODE__)
    message(STATUS "Build in GPU Mode.")
endif (CUDA_FOUND AND CUDA_FOUND AND START_CUDA)

set(PROJ_INCLUDE_DIR
        ${PROJECT_SOURCE_DIR}/include)

### Compile *.PTX code
if (BUILD_GPU_ENABLED)
    set(RENDER_PTX_LIB render_ptx_lib)
    set(OPTIX_RAYTRACING_FILE src/tunan/gpu/optix_ray.cu)
    add_library(${RENDER_PTX_LIB} OBJECT ${OPTIX_RAYTRACING_FILE})
    set_property(TARGET ${RENDER_PTX_LIB} PROPERTY CUDA_PTX_COMPILATION ON)
    target_compile_options("${RENDER_PTX_LIB}"
            PRIVATE
            -Xcudafe=--display_error_number -Xcudafe=--diag_suppress=3089)
    target_include_directories(${RENDER_PTX_LIB}
            PRIVATE
            ${CUDA_INCLUDE_DIRECTORY}
            ${OPTIX_INCLUDE_DIRECTORY}
            ${PROJ_INCLUDE_DIR})
    target_link_libraries(${RENDER_PTX_LIB}
            PRIVATE
            ${CUDA_LINK_LIBRARY})

    set(c_ptx_code_val_name "OptixPtxCode")
    set(OPTIX_EMBEDDED_FILE ${OPTIX_RAYTRACING_FILE}.ptx_embedded.cpp)
    add_custom_command(
            OUTPUT "${OPTIX_EMBEDDED_FILE}"
            COMMAND ${CMAKE_COMMAND}
            "-DBIN_TO_C_COMMAND=${BIN2C}"
            "-DOBJECTS=$<TARGET_OBJECTS:${RENDER_PTX_LIB}>"
            "-DVAR_NAME=${c_ptx_code_val_name}"
            "-DOUTPUT=${OPTIX_EMBEDDED_FILE}"
            -P ${CMAKE_MODULE_PATH}/bin2c_wrapper.cmake
            VERBATIM
            DEPENDS "${RENDER_PTX_LIB}" $<TARGET_OBJECTS:${RENDER_PTX_LIB}>
    )

    message(STATUS "Build *.PTX embedded code: ${OPTIX_EMBEDDED_FILE}")
    add_library(render_optix_ptxcode_lib STATIC ${OPTIX_EMBEDDED_FILE})

endif (BUILD_GPU_ENABLED)

set(RENDER_BASE_SRCS
        src/tunan/base/interactions.cpp
        include/tunan/base/spectrum.h
        include/tunan/base/containers.h
        include/tunan/base/mediums.h)

set(RENDER_UTILS_SRCS
        src/tunan/utils/MemoryAllocator.cpp
        src/tunan/utils/memory/CPUResource.cpp
        include/tunan/utils/type_utils.h
        include/tunan/utils/TaggedPointer.h)

set(RENDER_MATERIAL_SRCS
        src/tunan/material/Lambertian.cpp
        src/tunan/material/Material.cpp)

set(RENDER_MEDIA_SRCS
        include/tunan/medium/Medium.h)

set(RENDER_SCENE_SRCS
        include/tunan/common.h)

set(RENDER_SRCS
        main.cpp)

### Set cuda properties
if (BUILD_GPU_ENABLED)
    set(RENDER_UTILS_SRCS
            ${RENDER_UTILS_SRCS}
            src/tunan/utils/memory/CUDAResource.cpp)

    set(RENDER_SCENE_SRCS
            ${RENDER_SCENE_SRCS}
            src/tunan/scene/OptiXScene.cpp)

    set_source_files_properties(
            include/tunan/utils/TaggedPointer.h
            include/tunan/math.h
            ${RENDER_SRCS}
            ${RENDER_SCENE_SRCS}
            ${RENDER_BASE_SRCS}
            ${RENDER_MEDIA_SRCS}
            ${RENDER_MATERIAL_SRCS}
            PROPERTIES LANGUAGE CUDA)
endif ()

add_library(render_base_lib
        ${RENDER_BASE_SRCS})
target_include_directories(render_base_lib
        PRIVATE
        ${PROJ_INCLUDE_DIR})

add_library(render_media_lib
        ${RENDER_MEDIA_SRCS})
target_include_directories(render_media_lib
        PRIVATE
        ${PROJ_INCLUDE_DIR})

add_library(render_utils_lib
        ${RENDER_UTILS_SRCS})
target_include_directories(
        render_utils_lib
        PRIVATE
        ${CUDA_INCLUDE_DIRECTORY}
        ${PROJ_INCLUDE_DIR}
)
target_link_libraries(
        render_utils_lib
        PRIVATE
        ${CUDA_LINK_LIBRARY}
)

add_library(render_material_lib
        ${RENDER_MATERIAL_SRCS})
target_include_directories(
        render_material_lib
        PRIVATE
        ${PROJ_INCLUDE_DIR}
)
target_link_libraries(
        render_material_lib
        PRIVATE
        render_base_lib
        render_utils_lib
)

add_library(render_scene_lib
        ${RENDER_SCENE_SRCS})
target_include_directories(
        render_scene_lib
        PRIVATE
        ${PROJ_INCLUDE_DIR}
        ${OPTIX_INCLUDE_DIRECTORY}
        ${CUDA_INCLUDE_DIRECTORY}
)

target_link_libraries(
        render_scene_lib
        PRIVATE
        render_optix_ptxcode_lib
        ${CUDA_LINK_LIBRARY}
)

add_executable(${PROJECT_NAME} ${RENDER_SRCS})
target_link_libraries(
        ${PROJECT_NAME}
        PRIVATE
        render_scene_lib
        render_base_lib
        render_media_lib
        render_utils_lib
        render_material_lib
        ${CUDA_LINK_LIBRARY}
)
target_include_directories(
        ${PROJECT_NAME}
        PRIVATE
        ${OPTIX_INCLUDE_DIRECTORY}
        ${CUDA_INCLUDE_DIRECTORY}
        ${PROJ_INCLUDE_DIR}
)