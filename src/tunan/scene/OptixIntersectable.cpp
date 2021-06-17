//
// Created by StormPhoenix on 2021/5/31.
//

#include <tunan/common.h>
#include <tunan/base/containers.h>
#include <tunan/scene/cameras.h>
#include <tunan/scene/meshes.h>
#include <tunan/scene/OptixIntersectable.h>
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

    OptixIntersectable::OptixIntersectable(SceneData &sceneData, ResourceManager *allocator) :
            allocator(allocator), closestHitRecords(allocator), shadowRayRecords(allocator) {
        buildIntersectionStruct(sceneData);
    }

    void OptixIntersectable::initParams(SceneData &sceneData) {
        // Output image buffer
        uchar3 *deviceOutputBuffer = nullptr;
        {
            CUDA_CHECK(cudaFree(reinterpret_cast<void *>( deviceOutputBuffer )));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>( &deviceOutputBuffer ),
                                  sceneData.width * sceneData.height * sizeof(uchar3)));
        }
    }

    void OptixIntersectable::createContext() {
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
    }

    void OptixIntersectable::buildAccelStruct(SceneData &sceneData) {
        // Build traversable handle
        OptixTraversableHandle triangleGASHandler = createTriangleGAS(sceneData, state.closestHitPG,
                                                                      state.anyHitShadowRayPG);

        int sbtOffsetTriangles = 0;
        base::Vector <OptixInstance> iasInstances(allocator);

        OptixInstance gasInstance = {};
        float identity[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
        memcpy(gasInstance.transform, identity, 12 * sizeof(float));
        gasInstance.visibilityMask = 255;
        gasInstance.flags = OPTIX_INSTANCE_FLAG_NONE;
        if (triangleGASHandler != 0) {
            gasInstance.traversableHandle = triangleGASHandler;
            gasInstance.sbtOffset = sbtOffsetTriangles;
            iasInstances.push_back(gasInstance);
        }

        // Top level acceleration structure
        OptixBuildInput buildInput = {};
        buildInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
        buildInput.instanceArray.instances = CUdeviceptr(iasInstances.data());
        buildInput.instanceArray.numInstances = iasInstances.size();
        std::vector <OptixBuildInput> buildInputs = {buildInput};

        OptixTraversableHandle rootTraversable = buildBVH({buildInput});
        state.params.traversable = rootTraversable;
    }

    void OptixIntersectable::createModule() {
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
                    state.optixContext, &moduleCompileOptions, &pipelineCompileOptions,
                    ptxCode.c_str(), ptxCode.size(), log, &logSize, &optixModule), log);

        }
        state.pipelineCompileOptions = pipelineCompileOptions;
        state.optixModule = optixModule;
    }

    void OptixIntersectable::createProgramGroups() {
        char log[4096];
        size_t logSize = sizeof(log);

        OptixProgramGroupOptions pgOptions = {};
        OptixProgramGroup raygenFindClosestHitPG;
        OptixProgramGroup missFindClosestHitPG;
        {
            // Ray find intersection program group
            OptixProgramGroupDesc raygenFindIntersectionDesc = {};
            raygenFindIntersectionDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
            raygenFindIntersectionDesc.raygen.module = state.optixModule;
            raygenFindIntersectionDesc.raygen.entryFunctionName = "__raygen__findClosestHit";

            OPTIX_CHECK_LOG(optixProgramGroupCreate(
                    state.optixContext,
                    &raygenFindIntersectionDesc,
                    1, // num program groups
                    &pgOptions,
                    log, &logSize,
                    &raygenFindClosestHitPG), log);

            OptixProgramGroupDesc raygenMissingDesc = {};
            raygenMissingDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
            raygenMissingDesc.miss.module = state.optixModule;
            raygenMissingDesc.miss.entryFunctionName = "__miss__findclosesthit";
            OPTIX_CHECK_LOG(optixProgramGroupCreate(state.optixContext, &raygenMissingDesc, 1, /* num program groups */
                                                    &pgOptions, log, &logSize,
                                                    &missFindClosestHitPG), log);
        }
        state.raygenPG = raygenFindClosestHitPG;
        state.missPG = missFindClosestHitPG;

        OptixProgramGroup closeHitPG;
        {
            OptixProgramGroupDesc desc = {};
            desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
            desc.hitgroup.moduleCH = state.optixModule;
            desc.hitgroup.entryFunctionNameCH = "__closesthit__scene";
            desc.hitgroup.moduleAH = state.optixModule;
            desc.hitgroup.entryFunctionNameAH = "__anyhit__scene";
            OPTIX_CHECK_LOG(optixProgramGroupCreate(
                    state.optixContext, &desc, 1, &pgOptions,
                    log, &logSize, &closeHitPG), log);
        }
        state.closestHitPG = closeHitPG;

        OptixProgramGroup raygenShadowRayPG;
        {
            OptixProgramGroupDesc desc = {};
            desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
            desc.raygen.module = state.optixModule;
            desc.raygen.entryFunctionName = "__raygen__shadowRay";
            OPTIX_CHECK_LOG(optixProgramGroupCreate(
                    state.optixContext, &desc, 1, &pgOptions,
                    log, &logSize, &raygenShadowRayPG), log);
        }
        state.raygenShadowRayPG = raygenShadowRayPG;

        OptixProgramGroup anyHitShadowRayPG;
        {
            OptixProgramGroupDesc desc = {};
            desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
            desc.hitgroup.moduleAH = state.optixModule;
            desc.hitgroup.entryFunctionNameAH = "__anyhit__shadowRay";
            OPTIX_CHECK_LOG(optixProgramGroupCreate(
                    state.optixContext, &desc, 1, &pgOptions,
                    log, &logSize, &anyHitShadowRayPG), log);
        }
        state.anyHitShadowRayPG = anyHitShadowRayPG;

        OptixProgramGroup missShadowRayPG;
        {
            OptixProgramGroupDesc desc = {};
            desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
            desc.miss.module = state.optixModule;
            desc.miss.entryFunctionName = "__miss__shadowRay";
            OPTIX_CHECK_LOG(optixProgramGroupCreate(state.optixContext, &desc, 1, &pgOptions,
                                                    log, &logSize, &missShadowRayPG), log);
        }
        state.missShadowRayPG = missShadowRayPG;
    }

    void OptixIntersectable::createPipeline() {
        char log[4096];
        size_t logSize = sizeof(log);

        // Optix pipeline
        OptixPipeline pipeline = nullptr;
        {
            OptixPipelineLinkOptions pipelineLinkOptions = {};
            pipelineLinkOptions.maxTraceDepth = 2;
            // TODO check debug !!!
            pipelineLinkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

            OptixProgramGroup programGroups[] = {
                    state.raygenPG,
                    state.missPG,
                    state.closestHitPG,

                    state.raygenShadowRayPG,
                    state.missShadowRayPG,
            };

            OPTIX_CHECK_LOG(optixPipelineCreate(
                    state.optixContext,
                    &(state.pipelineCompileOptions),
                    &pipelineLinkOptions,
                    programGroups,
                    sizeof(programGroups) / sizeof(programGroups[0]),
                    log, &logSize,
                    &pipeline), log);
        }
        state.optixPipeline = pipeline;
    }

    void OptixIntersectable::createSBT() {
        // Shader binding table
        OptixShaderBindingTable closeHitSbt = {};
        {
            // Raygen record
            RaygenRecord hostRaygenRecord;
            OPTIX_CHECK(optixSbtRecordPackHeader(state.raygenPG, &hostRaygenRecord));
            // TODO test ragen data
            hostRaygenRecord.data = {0.462f};

            CUdeviceptr deviceRaygenRecord;
            const size_t raygenRecordSize = sizeof(RaygenRecord);
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>( &deviceRaygenRecord ), raygenRecordSize));
            CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>( deviceRaygenRecord ),
                                  &hostRaygenRecord, raygenRecordSize, cudaMemcpyHostToDevice));

            // Miss record
            MissRecord hostMissRecord;
            OPTIX_CHECK(optixSbtRecordPackHeader(state.missPG, &hostMissRecord));
            // TODO test miss data
            hostMissRecord.data = {0.74};

            CUdeviceptr deviceMissRecord;
            size_t missRecordSize = sizeof(MissRecord);
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>( &deviceMissRecord ), missRecordSize));
            CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>( deviceMissRecord ),
                                  &hostMissRecord, missRecordSize, cudaMemcpyHostToDevice));

            closeHitSbt.raygenRecord = deviceRaygenRecord;
            closeHitSbt.missRecordBase = deviceMissRecord;
            closeHitSbt.missRecordStrideInBytes = sizeof(MissRecord);
            closeHitSbt.missRecordCount = 1;

            // Hit group record
            closeHitSbt.hitgroupRecordBase = CUdeviceptr(closestHitRecords.data());
            closeHitSbt.hitgroupRecordStrideInBytes = sizeof(HitRecord);
            closeHitSbt.hitgroupRecordCount = closestHitRecords.size();
        }
        state.closeHitSbt = closeHitSbt;

        OptixShaderBindingTable shadowRaySbt = {};
        {
            // Raygen record
            RaygenRecord hostRaygenRecord;
            OPTIX_CHECK(optixSbtRecordPackHeader(state.raygenShadowRayPG, &hostRaygenRecord));
            // TODO test ragen data
            hostRaygenRecord.data = {0.462f};

            CUdeviceptr deviceRaygenRecord;
            const size_t raygenRecordSize = sizeof(RaygenRecord);
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>( &deviceRaygenRecord ), raygenRecordSize));
            CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>( deviceRaygenRecord ),
                                  &hostRaygenRecord, raygenRecordSize, cudaMemcpyHostToDevice));

            // Miss record
            MissRecord hostMissRecord;
            OPTIX_CHECK(optixSbtRecordPackHeader(state.missShadowRayPG, &hostMissRecord));
            // TODO test miss data
            hostMissRecord.data = {0.74};

            CUdeviceptr deviceMissRecord;
            size_t missRecordSize = sizeof(MissRecord);
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>( &deviceMissRecord ), missRecordSize));
            CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>( deviceMissRecord ),
                                  &hostMissRecord, missRecordSize, cudaMemcpyHostToDevice));

            shadowRaySbt.raygenRecord = deviceRaygenRecord;
            shadowRaySbt.missRecordBase = deviceMissRecord;
            shadowRaySbt.missRecordStrideInBytes = sizeof(MissRecord);
            shadowRaySbt.missRecordCount = 1;

            // Hit group record
            shadowRaySbt.hitgroupRecordBase = CUdeviceptr(shadowRayRecords.data());
            shadowRaySbt.hitgroupRecordStrideInBytes = sizeof(HitRecord);
            shadowRaySbt.hitgroupRecordCount = shadowRayRecords.size();
        }
        state.shadowRaySbt = shadowRaySbt;
    }

    void OptixIntersectable::buildIntersectionStruct(SceneData &sceneData) {
        initParams(sceneData);
        createContext();
        createModule();
        createProgramGroups();
        buildAccelStruct(sceneData);
        createPipeline();
        createSBT();
    }

    void
    OptixIntersectable::findClosestHit(RayQueue *rayQueue, MissQueue *missQueue, MaterialEvaQueue *materialEvaQueue,
                                       MediaEvaQueue *mediaEvaQueue, AreaLightHitQueue *areaLightQueue,
                                       PixelStateArray *pixelStateArray) {
        if (rayQueue->size() == 0) {
            return;
        }
        state.params.rayQueue = rayQueue;
        state.params.missQueue = missQueue;
        state.params.materialEvaQueue = materialEvaQueue;
        state.params.areaLightQueue = areaLightQueue;
        state.params.pixelStateArray = pixelStateArray;

        void *deviceParams;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>( &deviceParams ), sizeof(RayParams)));
        CUDA_CHECK(cudaMemcpy(
                reinterpret_cast<void *>( deviceParams ),
                &(state.params), sizeof(state.params),
                cudaMemcpyHostToDevice
        ));

        int nRayItem = rayQueue->size();
        OPTIX_CHECK(optixLaunch(state.optixPipeline, state.cudaStream, CUdeviceptr(deviceParams), sizeof(RayParams),
                                &state.closeHitSbt, /*width=*/nRayItem, /*height=*/1, /*depth=*/1));
        CUDA_SYNC_CHECK();
        CUDA_CHECK(cudaFree(deviceParams));
    }

    void OptixIntersectable::traceShadowRay(ShadowRayQueue *shadowRayQueue, PixelStateArray *pixelStateArray) {
        if (shadowRayQueue->size() == 0) {
            return;
        }
        state.params.shadowRayQueue = shadowRayQueue;
        state.params.pixelStateArray = pixelStateArray;

        void *deviceParams;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>( &deviceParams ), sizeof(RayParams)));
        CUDA_CHECK(cudaMemcpy(
                reinterpret_cast<void *>( deviceParams ),
                &(state.params), sizeof(state.params),
                cudaMemcpyHostToDevice));

        int nRayItem = shadowRayQueue->size();
        OPTIX_CHECK(optixLaunch(state.optixPipeline, state.cudaStream, CUdeviceptr(deviceParams), sizeof(RayParams),
                                &state.shadowRaySbt, /*width=*/nRayItem, /*height=*/1, /*depth=*/1));
        CUDA_SYNC_CHECK();
        CUDA_CHECK(cudaFree(deviceParams));
    }

    OptixTraversableHandle OptixIntersectable::createTriangleGAS(SceneData &data, OptixProgramGroup &closestHitPG,
                                                                 OptixProgramGroup &shadowRayHitPG) {
        std::vector < TriangleMesh * > meshes;
        std::vector <CUdeviceptr> devicePtrConversion;
        std::vector <uint32_t> triangleBuildInputFlag;

        size_t shapeCount = data.entities.size();
        meshes.resize(shapeCount);
        devicePtrConversion.resize(shapeCount);
        triangleBuildInputFlag.resize(shapeCount);

        // Create meshes
        for (int i = 0; i < shapeCount; i++) {
            ShapeEntity &entity = data.entities[i];
            meshes[i] = allocator->newObject<TriangleMesh>(
                    entity.nVertices, entity.vertices, entity.nNormals, entity.normals,
                    entity.nTexcoords, entity.texcoords, entity.nTriangles, entity.vertexIndices,
                    entity.normalIndices, entity.texcoordIndices, entity.toWorld);
        }

        std::vector <OptixBuildInput> buildInputs;
        buildInputs.resize(shapeCount);
        for (int i = 0; i < shapeCount; i++) {
            ShapeEntity &entity = data.entities[i];
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

            // Triangle mesh data are in object space by default
            input.triangleArray.transformFormat = OPTIX_TRANSFORM_FORMAT_NONE;
//            input.triangleArray.transformFormat = OPTIX_TRANSFORM_FORMAT_MATRIX_FLOAT12;
//            input.triangleArray.preTransform = CUdeviceptr(mesh->transformMatrix);

            // Indices
            input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
            input.triangleArray.indexStrideInBytes = 3 * sizeof(int);
            input.triangleArray.numIndexTriplets = mesh->nTriangles;
            input.triangleArray.indexBuffer = CUdeviceptr(mesh->vertexIndices);
            triangleBuildInputFlag[i] = OPTIX_GEOMETRY_FLAG_NONE;
            input.triangleArray.flags = &triangleBuildInputFlag[i];

            // SBT
            input.triangleArray.numSbtRecords = 1;
            input.triangleArray.sbtIndexOffsetBuffer = CUdeviceptr(0);
            input.triangleArray.sbtIndexOffsetSizeInBytes = 0;
            input.triangleArray.sbtIndexOffsetStrideInBytes = 0;

            buildInputs[i] = input;

            // Set shader binding table data
            HitRecord hitRecord;
            OPTIX_CHECK(optixSbtRecordPackHeader(closestHitPG, &hitRecord));
            hitRecord.data.mesh = mesh;
            hitRecord.data.material = entity.material;
            hitRecord.data.areaLights = entity.areaLights;
            closestHitRecords.push_back(hitRecord);

            OPTIX_CHECK(optixSbtRecordPackHeader(shadowRayHitPG, &hitRecord));
            shadowRayRecords.push_back(hitRecord);
        }

        if (!buildInputs.empty()) {
            return buildBVH(buildInputs);
        } else {
            return {};
        }
    }

    OptixTraversableHandle OptixIntersectable::buildBVH(const std::vector <OptixBuildInput> &buildInputs) {
        // Figure out memory requirements.
        OptixAccelBuildOptions accelOptions = {};
        accelOptions.buildFlags = (OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE);
        accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

        OptixAccelBufferSizes gasBufferSizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(state.optixContext, &accelOptions,
                                                 buildInputs.data(), buildInputs.size(),
                                                 &gasBufferSizes));
        uint64_t *compactedSizeBufferPtr = allocator->newObject<uint64_t>();

        OptixAccelEmitDesc emitDesc = {};
        emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emitDesc.result = (CUdeviceptr) compactedSizeBufferPtr;

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
                &emitDesc, 1));
        CUDA_CHECK(cudaStreamSynchronize(0));

        buildBVHBytes += *((uint64_t * )(emitDesc.result));

        // Compact
        void *asBuffer;
        CUDA_CHECK(cudaMalloc(&asBuffer, *compactedSizeBufferPtr));

        OPTIX_CHECK(optixAccelCompact(state.optixContext, state.cudaStream, traversableHandle,
                                      CUdeviceptr(asBuffer), *compactedSizeBufferPtr,
                                      &traversableHandle));
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaFree(deviceTempBuffer));
        CUDA_CHECK(cudaFree(deviceOutputBuffer));

        return traversableHandle;
    }
}