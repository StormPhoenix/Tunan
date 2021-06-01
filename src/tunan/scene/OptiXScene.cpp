//
// Created by StormPhoenix on 2021/5/31.
//

#include <tunan/common.h>
#include <tunan/scene/OptiXScene.h>
#include <tunan/gpu/cuda_utils.h>
#include <tunan/gpu/optix_utils.h>
#include <tunan/gpu/optix_ray.h>
#include <tunan/utils/image_utils.h>

#include <sstream>

#include <cuda.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

extern "C" {
extern const unsigned char OptixPtxCode[];
}

namespace RENDER_NAMESPACE {
    // TODO delete
    using namespace utils;
    // TODO for testing
    template<typename T>
    struct SbtRecord {
        __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        T data;
    };
    typedef SbtRecord<RayGenData> RayGenSbtRecord;
    typedef SbtRecord<int> MissSbtRecord;

    static void optiXLogCallback(unsigned int level, const char *tag, const char *message, void *cbdata) {
        std::cout << "OptiX callback: " << tag << ": " << message << std::endl;
    }

    void OptiXScene::buildOptiXData() {
        // Initialize optix context
        OptixDeviceContext optixContext;
        {
            // Initialize current cuda context
            CUDA_CHECK(cudaFree(0));
            CUcontext cudaContext = 0;
//            CU_CHECK(cuCtxGetCurrent(&cudaContext));
//            CHECK(cudaContext != nullptr);

            OPTIX_CHECK(optixInit());
            OptixDeviceContextOptions options = {};
            options.logCallbackFunction = &optiXLogCallback;
            options.logCallbackLevel = 4;
            OPTIX_CHECK(optixDeviceContextCreate(cudaContext, &options, &optixContext));
        }
        state.optixContext = optixContext;

        // log buffer
        char log[4096];
        size_t logSize = sizeof(log);

        // Create module
        OptixModule optixModule = nullptr;
        OptixPipelineCompileOptions pipelineCompileOptions = {};
        {
            // Module compile options
            OptixModuleCompileOptions moduleCompileOptions = {};
            moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
            moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
            moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

            pipelineCompileOptions.usesMotionBlur = false;
            pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;

            // TODO check !!!
            pipelineCompileOptions.numPayloadValues = 3;
            // TODO check !!!
            pipelineCompileOptions.numAttributeValues = 4;

            // OPTIX_EXCEPTION_FLAG_NONE;
            // TODO // Enables debug exceptions during optix launches. This may incur significant performance cost and
            //  should only be done during development.
            pipelineCompileOptions.exceptionFlags =
                    (OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW |
                     OPTIX_EXCEPTION_FLAG_TRACE_DEPTH);
            pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";

            // Create ptx code
            const std::string ptxCode = std::string((const char *) OptixPtxCode);

            OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
                    optixContext, &moduleCompileOptions, &pipelineCompileOptions,
                    ptxCode.c_str(), ptxCode.size(), log, &logSize, &optixModule), log);

        }
        state.optixModule = optixModule;

        // Create programs
        OptixProgramGroupOptions programGroupOptions = {};
        OptixProgramGroup raygenFindIntersectionGroup;
        OptixProgramGroup raygenMissingGroup;
        {
            // Ray find intersection program group
            OptixProgramGroupDesc raygenFindIntersectionDesc = {};
            raygenFindIntersectionDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
            raygenFindIntersectionDesc.raygen.module = state.optixModule;
            raygenFindIntersectionDesc.raygen.entryFunctionName = "__raygen__findIntersection";

            OPTIX_CHECK_LOG(optixProgramGroupCreate(
                    state.optixContext,
                    &raygenFindIntersectionDesc,
                    1, // num program groups
                    &programGroupOptions,
                    log, &logSize,
                    &raygenFindIntersectionGroup), log);

            // Ray missing program group
            // TODO temp for nullptr
            OptixProgramGroupDesc raygenMissingDesc = {};
            raygenMissingDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
            OPTIX_CHECK_LOG(optixProgramGroupCreate(
                    state.optixContext,
                    &raygenMissingDesc,
                    1, // num program groups
                    &programGroupOptions,
                    log, &logSize,
                    &raygenMissingGroup), log);
        }
        state.rayFindIntersectionGroup = raygenFindIntersectionGroup;
        state.rayMissingGroup = raygenMissingGroup;

        // Optix pipeline
        OptixPipeline pipeline = nullptr;
        {
            OptixPipelineLinkOptions pipelineLinkOptions = {};
            pipelineLinkOptions.maxTraceDepth = 2;
            // TODO check debug !!!
            pipelineLinkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

            OptixProgramGroup programGroups[] = {
                    state.rayFindIntersectionGroup,
                    state.rayMissingGroup
            };

            OPTIX_CHECK_LOG(optixPipelineCreate(
                    state.optixContext,
                    &pipelineCompileOptions,
                    &pipelineLinkOptions,
                    programGroups,
                    sizeof(programGroups) / sizeof(programGroups[0]),
                    log, &logSize,
                    &pipeline), log);
        }
        state.optixPipeline = pipeline;

        // Shader bingding table
        // TODO for testing
        OptixShaderBindingTable sbt = {};
        {
            CUdeviceptr raygen_record;
            const size_t raygen_record_size = sizeof(RayGenSbtRecord);
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>( &raygen_record ), raygen_record_size));
            RayGenSbtRecord rg_sbt;
            OPTIX_CHECK(optixSbtRecordPackHeader(state.rayFindIntersectionGroup, &rg_sbt));
            rg_sbt.data = {0.462f, 0.725f, 0.f};
            CUDA_CHECK(cudaMemcpy(
                    reinterpret_cast<void *>( raygen_record ),
                    &rg_sbt,
                    raygen_record_size,
                    cudaMemcpyHostToDevice
            ));

            CUdeviceptr miss_record;
            size_t miss_record_size = sizeof(MissSbtRecord);
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>( &miss_record ), miss_record_size));
            RayGenSbtRecord ms_sbt;
            OPTIX_CHECK(optixSbtRecordPackHeader(state.rayMissingGroup, &ms_sbt));
            CUDA_CHECK(cudaMemcpy(
                    reinterpret_cast<void *>( miss_record ),
                    &ms_sbt,
                    miss_record_size,
                    cudaMemcpyHostToDevice
            ));

            sbt.raygenRecord = raygen_record;
            sbt.missRecordBase = miss_record;
            sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
            sbt.missRecordCount = 1;
        }
        state.sbt = sbt;
    }

    void OptiXScene::intersect() {
        // TODO for test
        int width = 400;
        int height = 400;

        uchar3 *m_device_pixels = nullptr;
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>( m_device_pixels )));
        CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void **>( &m_device_pixels ),
                width * height * sizeof(uchar3)
        ));

        CUDA_CHECK(cudaStreamCreate(&(state.cudaStream)));
        RayParams params;
        params.image = m_device_pixels;
        params.width = width;
        params.height = height;

        CUdeviceptr d_param;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>( &d_param ), sizeof(RayParams)));
        CUDA_CHECK(cudaMemcpy(
                reinterpret_cast<void *>( d_param ),
                &params, sizeof(params),
                cudaMemcpyHostToDevice
        ));

        OPTIX_CHECK(optixLaunch(state.optixPipeline,
                                state.cudaStream,
                                d_param, sizeof(RayParams), &state.sbt, width, height, /*depth=*/1));
        CUDA_SYNC_CHECK();

        std::vector<uchar3> m_host_pixels;
        m_host_pixels.reserve(width * height);
        CUDA_CHECK(cudaMemcpy(
                static_cast<void *>(m_host_pixels.data()),
                params.image,
                params.width * params.height * sizeof(uchar3),
                cudaMemcpyDeviceToHost));

        // Write image
        writeImage("test.png", params.width, params.height, 3, m_host_pixels.data());
    }
}