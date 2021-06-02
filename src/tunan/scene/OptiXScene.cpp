//
// Created by StormPhoenix on 2021/5/31.
//

#include <tunan/common.h>
#include <tunan/base/containers.h>
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
    static void optiXLogCallback(unsigned int level, const char *tag, const char *message, void *cbdata) {
        std::cout << "OptiX callback: " << tag << ": " << message << std::endl;
    }

    void OptiXScene::buildOptiXData(const SceneData &sceneData) {
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

        OptixProgramGroupOptions programGroupOptions = {};
        // Raygen + missing program
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
        state.raygenFindIntersectionProgramGroup = raygenFindIntersectionGroup;
        state.missProgramGroup = raygenMissingGroup;

        // Closest hit program
        OptixProgramGroup closestHitProgramGroup;
        {
            OptixProgramGroupDesc desc = {};
            desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
            desc.hitgroup.moduleCH = state.optixModule;
            desc.hitgroup.entryFunctionNameCH = "__closesthit__scene";
            desc.hitgroup.moduleAH = optixModule;
            desc.hitgroup.entryFunctionNameAH = "__anyhit__scene";
            OPTIX_CHECK_LOG(optixProgramGroupCreate(
                    state.optixContext, &desc, 1, &programGroupOptions,
                    log, &logSize, &closestHitProgramGroup), log);
        }
        state.closestHitProgramGroup = closestHitProgramGroup;

        // Optix pipeline
        OptixPipeline pipeline = nullptr;
        {
            OptixPipelineLinkOptions pipelineLinkOptions = {};
            pipelineLinkOptions.maxTraceDepth = 2;
            // TODO check debug !!!
            pipelineLinkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

            OptixProgramGroup programGroups[] = {
                    state.raygenFindIntersectionProgramGroup,
                    state.missProgramGroup
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

        // Shader binding table
        OptixShaderBindingTable sbt = {};
        {
            // TODO for testing
            // Raygen record + Missing record
            CUdeviceptr raygen_record;
            const size_t raygen_record_size = sizeof(RaygenRecord);
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>( &raygen_record ), raygen_record_size));
            RaygenRecord rg_sbt;
            OPTIX_CHECK(optixSbtRecordPackHeader(state.raygenFindIntersectionProgramGroup, &rg_sbt));
            rg_sbt.data = {0.462f};
            CUDA_CHECK(cudaMemcpy(
                    reinterpret_cast<void *>( raygen_record ),
                    &rg_sbt,
                    raygen_record_size,
                    cudaMemcpyHostToDevice
            ));

            // TODO for testing
            CUdeviceptr miss_record;
            size_t miss_record_size = sizeof(RaygenRecord);
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>( &miss_record ), miss_record_size));
            RaygenRecord ms_sbt;
            OPTIX_CHECK(optixSbtRecordPackHeader(state.missProgramGroup, &ms_sbt));
            CUDA_CHECK(cudaMemcpy(
                    reinterpret_cast<void *>( miss_record ),
                    &ms_sbt,
                    miss_record_size,
                    cudaMemcpyHostToDevice
            ));

            sbt.raygenRecord = raygen_record;
            sbt.missRecordBase = miss_record;
            sbt.missRecordStrideInBytes = sizeof(RaygenRecord);
            sbt.missRecordCount = 1;
        }

        // Build traversable handle
        OptixTraversableHandle triangleGASHandler = createTriangleGAS(sceneData, state.closestHitProgramGroup);
        int triangleOffset = 0;

        // TODO self-define vector array
        base::Vector<OptixInstance> iasInstances(allocator);

        OptixInstance gasInstance = {};
        if (triangleGASHandler != 0) {
            gasInstance.traversableHandle = triangleGASHandler;
            gasInstance.sbtOffset = triangleOffset;
            iasInstances.push_back(gasInstance);
        }

        // Top level acceleration structure
        OptixBuildInput buildInput = {};
        buildInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
        buildInput.instanceArray.instances = CUdeviceptr(iasInstances.data());
        buildInput.instanceArray.numInstances = iasInstances.size();
        std::vector<OptixBuildInput> buildInputs = {buildInput};

        OptixTraversableHandle rootTraversable = buildBVH({buildInput});

        // Set closest hit sbt
        sbt.hitgroupRecordBase = CUdeviceptr(closestHitRecords.data());
        sbt.hitgroupRecordStrideInBytes = sizeof(ClosestHitRecord);
        sbt.hitgroupRecordCount = closestHitRecords.size();
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

        void *d_param;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>( &d_param ), sizeof(RayParams)));
        CUDA_CHECK(cudaMemcpy(
                reinterpret_cast<void *>( d_param ),
                &params, sizeof(params),
                cudaMemcpyHostToDevice
        ));

        OPTIX_CHECK(optixLaunch(state.optixPipeline,
                                state.cudaStream,
                                CUdeviceptr(d_param), sizeof(RayParams), &state.sbt, width, height, /*depth=*/1));
        CUDA_SYNC_CHECK();

        std::vector<uchar3> m_host_pixels;
        m_host_pixels.reserve(width * height);
        CUDA_CHECK(cudaMemcpy(
                static_cast<void *>(m_host_pixels.data()),
                params.image,
                params.width * params.height * sizeof(uchar3),
                cudaMemcpyDeviceToHost));

        // Write image
        utils::writeImage("test.png", params.width, params.height, 3, m_host_pixels.data());
        CUDA_CHECK(cudaFree(d_param));
    }

    OptixTraversableHandle OptiXScene::createTriangleGAS(const SceneData &data,
                                                         OptixProgramGroup &closestHitPG) {
        std::vector<TriangleMesh *> meshes;
        std::vector<CUdeviceptr> devicePtrConversion;
        std::vector<uint32_t> triangleBuildInputFlag;

        size_t shapeCount = data.entities.size();
        meshes.resize(shapeCount);
        devicePtrConversion.resize(shapeCount);
        triangleBuildInputFlag.resize(shapeCount);

        // Create meshes
        for (int i = 0; i < shapeCount; i++) {
            meshes[i] = allocator.newObject<TriangleMesh>(data.entities[i]);
        }

        std::vector<OptixBuildInput> buildInputs;
        buildInputs.resize(shapeCount);
        // TODO memcpy to device
        for (int i = 0; i < shapeCount; i++) {
            TriangleMesh *mesh = meshes[i];
            size_t nVertices = mesh->nVertices;

            OptixBuildInput input = {};
            input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
            // Vertices
            input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
            input.triangleArray.vertexStrideInBytes = sizeof(Point3F);
            input.triangleArray.numVertices = nVertices;
            devicePtrConversion[i] = CUdeviceptr(mesh->vertices);
            input.triangleArray.vertexBuffers = &(devicePtrConversion[i]);

            // Indices
            input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
            input.triangleArray.indexStrideInBytes = 3 * sizeof(int);
            input.triangleArray.numIndexTriplets = mesh->nTriangles;
            input.triangleArray.indexBuffer = CUdeviceptr(mesh->indices);
            triangleBuildInputFlag[i] = OPTIX_GEOMETRY_FLAG_NONE;
            input.triangleArray.flags = &triangleBuildInputFlag[i];

            // SBT
            input.triangleArray.numSbtRecords = 1;
            input.triangleArray.sbtIndexOffsetBuffer = CUdeviceptr(0);
            input.triangleArray.sbtIndexOffsetSizeInBytes = 0;
            input.triangleArray.sbtIndexOffsetStrideInBytes = 0;

            buildInputs[i] = input;

            // Set shader binding table data
            ClosestHitRecord hitRecord;
            OPTIX_CHECK(optixSbtRecordPackHeader(closestHitPG, &hitRecord));
            // TODO for testing
            hitRecord.data.r = 0.1;
            closestHitRecords.push_back(hitRecord);
        }

        if (!buildInputs.empty()) {
            return buildBVH(buildInputs);
        } else {
            return {};
        }
    }

    OptixTraversableHandle OptiXScene::buildBVH(std::vector<OptixBuildInput> buildInputs) {
        // Figure out memory requirements.
        OptixAccelBuildOptions accelOptions = {};
        accelOptions.buildFlags = (OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE);
        accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

        OptixAccelBufferSizes gasBufferSizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(state.optixContext, &accelOptions,
                                                 buildInputs.data(), buildInputs.size(),
                                                 &gasBufferSizes));

//        uint64_t *compactedSizeBufferPtr = allocator.newObject<uint64_t>();
        uint64_t *compactedSizeBufferPtr;
        CUDA_CHECK(cudaMalloc(&compactedSizeBufferPtr, sizeof(uint64_t)));

        OptixAccelEmitDesc emitProperty = {};
        emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emitProperty.result = (CUdeviceptr) compactedSizeBufferPtr;

        void *deviceTempBuffer;
        CUDA_CHECK(cudaMalloc(&deviceTempBuffer, gasBufferSizes.tempSizeInBytes));
        void *deviceOutputBuffer;
        CUDA_CHECK(cudaMalloc(&deviceOutputBuffer, gasBufferSizes.outputSizeInBytes));

        // Build.
        OptixTraversableHandle traversableHandle{0};
        OPTIX_CHECK(optixAccelBuild(
                state.optixContext, state.cudaStream, &accelOptions, buildInputs.data(), buildInputs.size(),
                CUdeviceptr(deviceTempBuffer), gasBufferSizes.tempSizeInBytes,
                CUdeviceptr(deviceOutputBuffer), gasBufferSizes.outputSizeInBytes, &traversableHandle,
                &emitProperty, 1));

        CUDA_CHECK(cudaDeviceSynchronize());

        buildBVHBytes += *compactedSizeBufferPtr;

        // Compact
        void *asBuffer;
        CUDA_CHECK(cudaMalloc(&asBuffer, *compactedSizeBufferPtr));

        OPTIX_CHECK(optixAccelCompact(state.optixContext, state.cudaStream, traversableHandle,
                                      CUdeviceptr(asBuffer), *compactedSizeBufferPtr,
                                      &traversableHandle));
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaFree(deviceTempBuffer));
        CUDA_CHECK(cudaFree(deviceOutputBuffer));
        CUDA_CHECK(cudaFree(compactedSizeBufferPtr));

        return traversableHandle;
    }
}