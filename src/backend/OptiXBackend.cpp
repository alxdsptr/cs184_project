#include "backend/OptiXBackend.h"
#include "backend/OptiXLaunchParams.h"
#include "backend/OptiXSBTRecords.h"
#include "scene/Scene.h"
#include "util/Log.h"
#include "util/CudaCheck.h"

#include <optix.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>
#include <optix_function_table_definition.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <vector>

#define OPTIX_CHECK(call)                                                      \
    do {                                                                       \
        OptixResult r = call;                                                  \
        if (r != OPTIX_SUCCESS) {                                              \
            LOG_ERROR("OptiX call failed (%d) at %s:%d: %s",                   \
                      (int)r, __FILE__, __LINE__, #call);                      \
            return false;                                                      \
        }                                                                      \
    } while (0)

#define OPTIX_CHECK_VOID(call)                                                 \
    do {                                                                       \
        OptixResult r = call;                                                  \
        if (r != OPTIX_SUCCESS) {                                              \
            LOG_ERROR("OptiX call failed (%d) at %s:%d: %s",                   \
                      (int)r, __FILE__, __LINE__, #call);                      \
        }                                                                      \
    } while (0)

static void optixLogCallback(unsigned int level, const char* tag, const char* msg, void*) {
    if (level <= 2)      LOG_ERROR("[OptiX %s] %s", tag ? tag : "", msg ? msg : "");
    else if (level == 3) LOG_WARN ("[OptiX %s] %s", tag ? tag : "", msg ? msg : "");
    else                 LOG_INFO ("[OptiX %s] %s", tag ? tag : "", msg ? msg : "");
}

OptiXBackend::OptiXBackend() {
    std::memset(&m_sbt, 0, sizeof(m_sbt));
}

OptiXBackend::~OptiXBackend() {
    destroyAll();
}

void OptiXBackend::destroyAll() {
    if (m_dLaunchParams) { cudaFree((void*)m_dLaunchParams); m_dLaunchParams = 0; }
    freeGAS();
    if (m_sbtRecordsBuf) { cudaFree((void*)m_sbtRecordsBuf); m_sbtRecordsBuf = 0; }
    if (m_pipeline)       { optixPipelineDestroy(m_pipeline); m_pipeline = nullptr; }
    if (m_pgRaygen)       { optixProgramGroupDestroy(m_pgRaygen); m_pgRaygen = nullptr; }
    if (m_pgMissRadiance) { optixProgramGroupDestroy(m_pgMissRadiance); m_pgMissRadiance = nullptr; }
    if (m_pgMissShadow)   { optixProgramGroupDestroy(m_pgMissShadow); m_pgMissShadow = nullptr; }
    if (m_pgHitRadiance)  { optixProgramGroupDestroy(m_pgHitRadiance); m_pgHitRadiance = nullptr; }
    if (m_pgHitShadow)    { optixProgramGroupDestroy(m_pgHitShadow); m_pgHitShadow = nullptr; }
    if (m_module)         { optixModuleDestroy(m_module); m_module = nullptr; }
    if (m_ctx)            { optixDeviceContextDestroy(m_ctx); m_ctx = nullptr; }
    if (m_stream)         { cudaStreamDestroy(m_stream); m_stream = nullptr; }
    m_deviceScene.free();
    m_initialized = false;
}

void OptiXBackend::freeGAS() {
    if (m_gasOutput) { cudaFree((void*)m_gasOutput); m_gasOutput = 0; }
    m_gasHandle = 0;
}

static bool readFile(const std::string& path, std::vector<char>& out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    std::streamsize sz = f.tellg();
    if (sz <= 0) return false;
    f.seekg(0, std::ios::beg);
    out.resize((size_t)sz);
    return !!f.read(out.data(), sz);
}

bool OptiXBackend::init(const std::string& optixirPath) {
    // Ensure a CUDA context exists (runtime API creates one lazily).
    cudaFree(nullptr);

    // Diagnose which stage of optixInit fails.
#ifdef _WIN32
    HMODULE hDll = LoadLibraryA("nvoptix.dll");
    if (!hDll) {
        DWORD err = GetLastError();
        LOG_ERROR("LoadLibraryA(\"nvoptix.dll\") failed, GetLastError=%lu", (unsigned long)err);
        // Try absolute driver store path fallback.
        const char* driverStore = "C:\\Windows\\System32\\DriverStore\\FileRepository\\nvtf.inf_amd64_f7df59a98b1aeb40\\nvoptix.dll";
        hDll = LoadLibraryA(driverStore);
        if (!hDll) {
            LOG_ERROR("LoadLibraryA(driverStore) also failed, GetLastError=%lu", (unsigned long)GetLastError());
        } else {
            LOG_INFO("LoadLibraryA succeeded from DriverStore path");
        }
    } else {
        LOG_INFO("LoadLibraryA(\"nvoptix.dll\") succeeded");
    }
#endif

    OPTIX_CHECK(optixInit());

    CUcontext cuCtx = nullptr;  // zero means use current context
    OptixDeviceContextOptions ctxOpts{};
    ctxOpts.logCallbackFunction = optixLogCallback;
    ctxOpts.logCallbackLevel    = 4;
#ifndef NDEBUG
    ctxOpts.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#endif
    OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &ctxOpts, &m_ctx));

    CUDA_CHECK(cudaStreamCreate(&m_stream));

    CUDA_CHECK(cudaMalloc((void**)&m_dLaunchParams, sizeof(LaunchParams)));

    if (!loadModule(optixirPath)) return false;
    if (!buildPipeline())          return false;
    if (!buildSBT())               return false;

    m_initialized = true;
    LOG_INFO("OptiXBackend: initialized");
    return true;
}

bool OptiXBackend::loadModule(const std::string& optixirPath) {
    std::vector<char> bytecode;
    if (!readFile(optixirPath, bytecode)) {
        LOG_ERROR("OptiXBackend: failed to open %s", optixirPath.c_str());
        return false;
    }

    OptixModuleCompileOptions moduleCompileOptions{};
    moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
#ifdef NDEBUG
    moduleCompileOptions.optLevel   = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
#else
    moduleCompileOptions.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif

    OptixPipelineCompileOptions pipelineCompileOptions{};
    pipelineCompileOptions.usesMotionBlur        = 0;
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipelineCompileOptions.numPayloadValues      = 5;  // radiance + shadow (max 5)
    pipelineCompileOptions.numAttributeValues    = 2;  // triangle barycentrics
    pipelineCompileOptions.exceptionFlags        = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";
    pipelineCompileOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

    char   logBuf[2048];
    size_t logSize = sizeof(logBuf);
    OptixResult r = optixModuleCreate(
        m_ctx, &moduleCompileOptions, &pipelineCompileOptions,
        bytecode.data(), bytecode.size(),
        logBuf, &logSize,
        &m_module);
    if (logSize > 1) LOG_INFO("OptiX module log: %s", logBuf);
    if (r != OPTIX_SUCCESS) {
        LOG_ERROR("optixModuleCreate failed: %d", (int)r);
        return false;
    }

    // Save pipeline compile options for buildPipeline.
    m_pipelineCompileOptions = pipelineCompileOptions;

    // Create program groups.
    OptixProgramGroupOptions pgOptions{};

    OptixProgramGroupDesc pgdRaygen{};
    pgdRaygen.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgdRaygen.raygen.module = m_module;
    pgdRaygen.raygen.entryFunctionName = "__raygen__path_trace";
    logSize = sizeof(logBuf);
    OPTIX_CHECK(optixProgramGroupCreate(m_ctx, &pgdRaygen, 1, &pgOptions, logBuf, &logSize, &m_pgRaygen));

    OptixProgramGroupDesc pgdMissR{};
    pgdMissR.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pgdMissR.miss.module = m_module;
    pgdMissR.miss.entryFunctionName = "__miss__radiance";
    logSize = sizeof(logBuf);
    OPTIX_CHECK(optixProgramGroupCreate(m_ctx, &pgdMissR, 1, &pgOptions, logBuf, &logSize, &m_pgMissRadiance));

    OptixProgramGroupDesc pgdMissS{};
    pgdMissS.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pgdMissS.miss.module = m_module;
    pgdMissS.miss.entryFunctionName = "__miss__shadow";
    logSize = sizeof(logBuf);
    OPTIX_CHECK(optixProgramGroupCreate(m_ctx, &pgdMissS, 1, &pgOptions, logBuf, &logSize, &m_pgMissShadow));

    OptixProgramGroupDesc pgdHitR{};
    pgdHitR.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pgdHitR.hitgroup.moduleCH = m_module;
    pgdHitR.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
    logSize = sizeof(logBuf);
    OPTIX_CHECK(optixProgramGroupCreate(m_ctx, &pgdHitR, 1, &pgOptions, logBuf, &logSize, &m_pgHitRadiance));

    OptixProgramGroupDesc pgdHitS{};
    pgdHitS.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pgdHitS.hitgroup.moduleAH = m_module;
    pgdHitS.hitgroup.entryFunctionNameAH = "__anyhit__shadow";
    logSize = sizeof(logBuf);
    OPTIX_CHECK(optixProgramGroupCreate(m_ctx, &pgdHitS, 1, &pgOptions, logBuf, &logSize, &m_pgHitShadow));

    return true;
}

bool OptiXBackend::buildPipeline() {
    OptixProgramGroup groups[] = {
        m_pgRaygen, m_pgMissRadiance, m_pgMissShadow,
        m_pgHitRadiance, m_pgHitShadow
    };

    OptixPipelineLinkOptions linkOptions{};
    linkOptions.maxTraceDepth = 2;  // primary trace can fire shadow trace

    char   logBuf[2048];
    size_t logSize = sizeof(logBuf);
    OPTIX_CHECK(optixPipelineCreate(
        m_ctx, &m_pipelineCompileOptions, &linkOptions,
        groups, (unsigned int)(sizeof(groups) / sizeof(groups[0])),
        logBuf, &logSize,
        &m_pipeline));

    OptixStackSizes stackSizes{};
    for (auto g : groups) {
        OPTIX_CHECK(optixUtilAccumulateStackSizes(g, &stackSizes, m_pipeline));
    }
    uint32_t directCallableFromTraversal = 0;
    uint32_t directCallableFromState     = 0;
    uint32_t continuationStack           = 0;
    OPTIX_CHECK(optixUtilComputeStackSizes(
        &stackSizes,
        /*maxTraceDepth*/2,
        /*maxCCDepth*/   0,
        /*maxDCDepth*/   0,
        &directCallableFromTraversal,
        &directCallableFromState,
        &continuationStack));
    OPTIX_CHECK(optixPipelineSetStackSize(
        m_pipeline,
        directCallableFromTraversal,
        directCallableFromState,
        continuationStack,
        /*maxTraversableGraphDepth*/1));
    return true;
}

bool OptiXBackend::buildSBT() {
    // Layout: [raygen][missR][missS][hitR][hitS]
    const size_t recSize = sizeof(RaygenRecord);  // all records same size (empty data)
    const size_t total   = recSize * 5;

    CUDA_CHECK(cudaMalloc((void**)&m_sbtRecordsBuf, total));

    RaygenRecord raygenRec;
    MissRecord   missR;
    MissRecord   missS;
    HitRecord_   hitR;
    HitRecord_   hitS;
    OPTIX_CHECK(optixSbtRecordPackHeader(m_pgRaygen,       &raygenRec));
    OPTIX_CHECK(optixSbtRecordPackHeader(m_pgMissRadiance, &missR));
    OPTIX_CHECK(optixSbtRecordPackHeader(m_pgMissShadow,   &missS));
    OPTIX_CHECK(optixSbtRecordPackHeader(m_pgHitRadiance,  &hitR));
    OPTIX_CHECK(optixSbtRecordPackHeader(m_pgHitShadow,    &hitS));

    char* base = (char*)m_sbtRecordsBuf;
    CUDA_CHECK(cudaMemcpy(base + 0 * recSize, &raygenRec, recSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(base + 1 * recSize, &missR,     recSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(base + 2 * recSize, &missS,     recSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(base + 3 * recSize, &hitR,      recSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(base + 4 * recSize, &hitS,      recSize, cudaMemcpyHostToDevice));

    std::memset(&m_sbt, 0, sizeof(m_sbt));
    m_sbt.raygenRecord                = (CUdeviceptr)(base + 0 * recSize);
    m_sbt.missRecordBase              = (CUdeviceptr)(base + 1 * recSize);
    m_sbt.missRecordStrideInBytes     = (unsigned int)recSize;
    m_sbt.missRecordCount             = 2;
    m_sbt.hitgroupRecordBase          = (CUdeviceptr)(base + 3 * recSize);
    m_sbt.hitgroupRecordStrideInBytes = (unsigned int)recSize;
    m_sbt.hitgroupRecordCount         = 2;
    return true;
}

bool OptiXBackend::buildGAS(const DeviceSceneData& data) {
    freeGAS();
    if (data.totalTriangles == 0) {
        LOG_WARN("OptiXBackend: no triangles to build GAS");
        return true;
    }

    unsigned int triangleFlags = OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL;

    OptixBuildInput buildInput{};
    buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    buildInput.triangleArray.vertexStrideInBytes = (unsigned int)sizeof(float3);
    buildInput.triangleArray.numVertices = data.totalVertices;
    CUdeviceptr d_positions = (CUdeviceptr)data.d_positions;
    buildInput.triangleArray.vertexBuffers = &d_positions;

    buildInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    buildInput.triangleArray.indexStrideInBytes = (unsigned int)(3 * sizeof(uint32_t));
    buildInput.triangleArray.numIndexTriplets = data.totalTriangles;
    buildInput.triangleArray.indexBuffer = (CUdeviceptr)data.d_indices;

    buildInput.triangleArray.flags = &triangleFlags;
    buildInput.triangleArray.numSbtRecords = 1;

    OptixAccelBuildOptions accelOptions{};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    accelOptions.operation  = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes bufSizes{};
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        m_ctx, &accelOptions, &buildInput, 1, &bufSizes));

    CUdeviceptr tempBuf = 0;
    CUdeviceptr uncompactedOutput = 0;
    CUDA_CHECK(cudaMalloc((void**)&tempBuf, bufSizes.tempSizeInBytes));
    CUDA_CHECK(cudaMalloc((void**)&uncompactedOutput, bufSizes.outputSizeInBytes));

    CUdeviceptr d_compactedSize = 0;
    CUDA_CHECK(cudaMalloc((void**)&d_compactedSize, sizeof(size_t)));
    OptixAccelEmitDesc emitDesc{};
    emitDesc.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitDesc.result = d_compactedSize;

    OptixTraversableHandle uncompactedHandle = 0;
    OPTIX_CHECK(optixAccelBuild(
        m_ctx, m_stream, &accelOptions, &buildInput, 1,
        tempBuf, bufSizes.tempSizeInBytes,
        uncompactedOutput, bufSizes.outputSizeInBytes,
        &uncompactedHandle, &emitDesc, 1));
    CUDA_CHECK(cudaStreamSynchronize(m_stream));

    size_t compactedSize = 0;
    CUDA_CHECK(cudaMemcpy(&compactedSize, (void*)d_compactedSize, sizeof(size_t), cudaMemcpyDeviceToHost));

    if (compactedSize < bufSizes.outputSizeInBytes) {
        CUDA_CHECK(cudaMalloc((void**)&m_gasOutput, compactedSize));
        OPTIX_CHECK(optixAccelCompact(
            m_ctx, m_stream, uncompactedHandle, m_gasOutput, compactedSize, &m_gasHandle));
        CUDA_CHECK(cudaStreamSynchronize(m_stream));
        cudaFree((void*)uncompactedOutput);
    } else {
        m_gasOutput = uncompactedOutput;
        m_gasHandle = uncompactedHandle;
    }

    cudaFree((void*)tempBuf);
    cudaFree((void*)d_compactedSize);

    LOG_INFO("OptiXBackend: GAS built (%u tris, compacted=%zu bytes)",
             data.totalTriangles, compactedSize);
    return true;
}

void OptiXBackend::buildAccelerationStructure(const Scene& scene) {
    if (!m_initialized) {
        LOG_ERROR("OptiXBackend: buildAccelerationStructure called before init");
        return;
    }
    // Upload mesh/material data. No reorder: OptiX preserves build-input order
    // so optixGetPrimitiveIndex() indexes d_materialIndices / d_triangleAreaLightIndex directly.
    m_deviceScene.upload(scene);
    DeviceSceneData data = m_deviceScene.getData();
    buildGAS(data);
}

void OptiXBackend::launchPathTrace(
    const DeviceSceneData& scene,
    const CameraParams& camera,
    float4* d_accumBuffer,
    float4* d_outputBuffer,
    AuxBufferPtrs auxBuffers,
    uint32_t width, uint32_t height,
    uint32_t sampleIndex,
    bool enableEnvironment,
    uint32_t maxBounces,
    uint32_t samplesPerPixel,
    PrimaryHitSurfaces gbufferSurfaces)
{
    if (!m_initialized || !m_gasHandle) return;

    LaunchParams lp;
    std::memset(&lp, 0, sizeof(lp));
    lp.scene   = scene;
    lp.camera  = camera;
    lp.accum   = d_accumBuffer;
    lp.output  = d_outputBuffer;
    lp.aux     = auxBuffers;
    lp.gbuffer = gbufferSurfaces;
    lp.width   = width;
    lp.height  = height;
    lp.sampleIndex = sampleIndex;
    lp.maxBounces  = maxBounces;
    lp.spp         = samplesPerPixel < 1 ? 1 : samplesPerPixel;
    lp.enableEnvironment = enableEnvironment ? 1u : 0u;
    lp.handle  = m_gasHandle;

    CUDA_CHECK(cudaMemcpyAsync(
        (void*)m_dLaunchParams, &lp, sizeof(lp),
        cudaMemcpyHostToDevice, m_stream));

    OPTIX_CHECK_VOID(optixLaunch(
        m_pipeline, m_stream,
        m_dLaunchParams, sizeof(LaunchParams),
        &m_sbt,
        width, height, 1));
    CUDA_CHECK(cudaStreamSynchronize(m_stream));
}

void OptiXBackend::traceOcclusionRays(
    const float3* /*d_origins*/,
    const float3* /*d_targets*/,
    bool* /*d_visible*/,
    uint32_t /*rayCount*/)
{
    // BDPT stub — unused currently.
}
