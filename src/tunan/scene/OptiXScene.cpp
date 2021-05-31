//
// Created by StormPhoenix on 2021/5/31.
//

#include <tunan/common.h>
#include <tunan/scene/OptiXScene.h>
#include <tunan/gpu/cuda_utils.h>
#include <tunan/gpu/optix_utils.h>

#include <sstream>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

extern "C" {
extern const unsigned char OptixPtxCode[];
}

namespace RENDER_NAMESPACE {

    static void optiXLogCallback(unsigned int level, const char *tag, const char *message, void *cbdata) {
        std::cout << "OptiX callback: " << tag << ": " << message << std::endl;
    }

    void OptiXScene::buildOptiXData() {
        // Initialize optix context
        OptixDeviceContext optixContext;
        {
            // Initialize current cuda context
            CUcontext cudaContext;
            CU_CHECK(cuCtxGetCurrent(&cudaContext));
            CHECK(cudaContext != nullptr);

            OPTIX_CHECK(optixInit());
            OptixDeviceContextOptions options = {};
            options.logCallbackFunction = &optiXLogCallback;
            options.logCallbackLevel = 4;
            OPTIX_CHECK(optixDeviceContextCreate(cudaContext, &options, &optixContext));
        }
        state.optixContext = optixContext;

        // Create module
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
            char log[4096];
            size_t logSize = sizeof(log);

            OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
                    optixContext, &moduleCompileOptions, &pipelineCompileOptions,
                    ptxCode.c_str(), ptxCode.size(), log, &logSize, &(state.optixModule)), log);

        }

        OptixPipelineLinkOptions pipelineLinkOptions = {};
        pipelineLinkOptions.maxTraceDepth = 2;
        // TODO check debug !!!
        pipelineLinkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

        // Create programs
        // TODO
    }
}