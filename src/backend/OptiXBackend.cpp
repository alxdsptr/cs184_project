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
    if (m_pgRaygenSplit)  { optixProgramGroupDestroy(m_pgRaygenSplit); m_pgRaygenSplit = nullptr; }
    if (m_pgRaygenReSTIR) { optixProgramGroupDestroy(m_pgRaygenReSTIR); m_pgRaygenReSTIR = nullptr; }
    if (m_pgRaygenReSTIRVis) { optixProgramGroupDestroy(m_pgRaygenReSTIRVis); m_pgRaygenReSTIRVis = nullptr; }
    if (m_pgRaygenReSTIRGI) { optixProgramGroupDestroy(m_pgRaygenReSTIRGI); m_pgRaygenReSTIRGI = nullptr; }
    if (m_pgRaygenReSTIRPT) { optixProgramGroupDestroy(m_pgRaygenReSTIRPT); m_pgRaygenReSTIRPT = nullptr; }
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

#ifdef PATHTRACER_NRD_DLSS_ENABLED
    // Second raygen for NRD's split-output path trace. Same hit/miss programs;
    // only the entry function differs. We pick which raygen to launch by
    // swapping m_sbt.raygenRecord at launch time.
    OptixProgramGroupDesc pgdRaygenSplit{};
    pgdRaygenSplit.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgdRaygenSplit.raygen.module = m_module;
    pgdRaygenSplit.raygen.entryFunctionName = "__raygen__path_trace_split";
    logSize = sizeof(logBuf);
    OPTIX_CHECK(optixProgramGroupCreate(m_ctx, &pgdRaygenSplit, 1, &pgOptions, logBuf, &logSize, &m_pgRaygenSplit));
#endif

    // Third raygen: ReSTIR DI initial-candidates pass. Shares the same
    // closest-hit / miss programs as __raygen__path_trace (they just record
    // the hit primitive + barycentrics into the 5-slot payload), so no new
    // hitgroup needed. Always built — enabling/disabling ReSTIR is a
    // runtime knob on Renderer, not a compile-time one.
    OptixProgramGroupDesc pgdRaygenReSTIR{};
    pgdRaygenReSTIR.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgdRaygenReSTIR.raygen.module = m_module;
    pgdRaygenReSTIR.raygen.entryFunctionName = "__raygen__restir_init_candidates";
    logSize = sizeof(logBuf);
    OPTIX_CHECK(optixProgramGroupCreate(m_ctx, &pgdRaygenReSTIR, 1, &pgOptions, logBuf, &logSize, &m_pgRaygenReSTIR));

    // Fourth raygen: ReSTIR DI visibility-reuse pass. One shadow ray per
    // pixel against the GAS to kill occluded samples before spatial /
    // temporal reuse propagates them. Shares the shadow miss + anyhit SBT
    // slots with the main path tracer.
    OptixProgramGroupDesc pgdRaygenReSTIRVis{};
    pgdRaygenReSTIRVis.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgdRaygenReSTIRVis.raygen.module = m_module;
    pgdRaygenReSTIRVis.raygen.entryFunctionName = "__raygen__restir_visibility";
    logSize = sizeof(logBuf);
    OPTIX_CHECK(optixProgramGroupCreate(m_ctx, &pgdRaygenReSTIRVis, 1, &pgOptions, logBuf, &logSize, &m_pgRaygenReSTIRVis));

    // Fifth raygen: ReSTIR GI initial-candidates pass. Casts primary +
    // indirect rays via the radiance hitgroup, plus one NEE shadow ray at
    // the indirect bounce. No new SBT slots needed.
    OptixProgramGroupDesc pgdRaygenReSTIRGI{};
    pgdRaygenReSTIRGI.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgdRaygenReSTIRGI.raygen.module = m_module;
    pgdRaygenReSTIRGI.raygen.entryFunctionName = "__raygen__restir_gi_init_candidates";
    logSize = sizeof(logBuf);
    OPTIX_CHECK(optixProgramGroupCreate(m_ctx, &pgdRaygenReSTIRGI, 1, &pgOptions, logBuf, &logSize, &m_pgRaygenReSTIRGI));

    // Sixth raygen: ReSTIR PT initial-candidates pass. Same SBT geometry as
    // ReSTIR GI (radiance + shadow program groups); the only difference is
    // the multi-bounce random walk past the reconnection vertex.
    OptixProgramGroupDesc pgdRaygenReSTIRPT{};
    pgdRaygenReSTIRPT.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgdRaygenReSTIRPT.raygen.module = m_module;
    pgdRaygenReSTIRPT.raygen.entryFunctionName = "__raygen__restir_pt_init_candidates";
    logSize = sizeof(logBuf);
    OPTIX_CHECK(optixProgramGroupCreate(m_ctx, &pgdRaygenReSTIRPT, 1, &pgOptions, logBuf, &logSize, &m_pgRaygenReSTIRPT));

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
    // Linker must see every program group whose record may sit in the SBT —
    // including the optional split raygen.
    std::vector<OptixProgramGroup> groups = {
        m_pgRaygen, m_pgMissRadiance, m_pgMissShadow,
        m_pgHitRadiance, m_pgHitShadow
    };
#ifdef PATHTRACER_NRD_DLSS_ENABLED
    if (m_pgRaygenSplit) groups.push_back(m_pgRaygenSplit);
#endif
    if (m_pgRaygenReSTIR) groups.push_back(m_pgRaygenReSTIR);
    if (m_pgRaygenReSTIRVis) groups.push_back(m_pgRaygenReSTIRVis);
    if (m_pgRaygenReSTIRGI) groups.push_back(m_pgRaygenReSTIRGI);
    if (m_pgRaygenReSTIRPT) groups.push_back(m_pgRaygenReSTIRPT);

    OptixPipelineLinkOptions linkOptions{};
    linkOptions.maxTraceDepth = 2;  // primary trace can fire shadow trace

    char   logBuf[2048];
    size_t logSize = sizeof(logBuf);
    OPTIX_CHECK(optixPipelineCreate(
        m_ctx, &m_pipelineCompileOptions, &linkOptions,
        groups.data(), (unsigned int)groups.size(),
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
    // Layout: [raygen][raygenSplit?][raygenReSTIR?][raygenReSTIRVis?][missR][missS][hitR][hitS]
    // raygenSplit is included only when NRD/DLSS is compiled in; the two
    // raygenReSTIR slots are always included (runtime-toggled). Launches pick
    // a raygen by rewriting m_sbt.raygenRecord to the appropriate slot.
    const size_t recSize = sizeof(RaygenRecord);  // all records same size (empty data)
#ifdef PATHTRACER_NRD_DLSS_ENABLED
    const bool   haveSplit = (m_pgRaygenSplit != nullptr);
#else
    const bool   haveSplit = false;
#endif
    const bool   haveReSTIR    = (m_pgRaygenReSTIR    != nullptr);
    const bool   haveReSTIRVis = (m_pgRaygenReSTIRVis != nullptr);
    const bool   haveReSTIRGI  = (m_pgRaygenReSTIRGI  != nullptr);
    const bool   haveReSTIRPT  = (m_pgRaygenReSTIRPT  != nullptr);
    const uint32_t numRaygens = 1u + (haveSplit ? 1u : 0u)
                                   + (haveReSTIR ? 1u : 0u)
                                   + (haveReSTIRVis ? 1u : 0u)
                                   + (haveReSTIRGI ? 1u : 0u)
                                   + (haveReSTIRPT ? 1u : 0u);
    const size_t total = recSize * (numRaygens + 4u);

    CUDA_CHECK(cudaMalloc((void**)&m_sbtRecordsBuf, total));

    RaygenRecord raygenRec;
    RaygenRecord raygenSplitRec;
    RaygenRecord raygenReSTIRRec;
    RaygenRecord raygenReSTIRVisRec;
    RaygenRecord raygenReSTIRGIRec;
    RaygenRecord raygenReSTIRPTRec;
    MissRecord   missR;
    MissRecord   missS;
    HitRecord_   hitR;
    HitRecord_   hitS;
    OPTIX_CHECK(optixSbtRecordPackHeader(m_pgRaygen,       &raygenRec));
    if (haveSplit) {
        OPTIX_CHECK(optixSbtRecordPackHeader(m_pgRaygenSplit, &raygenSplitRec));
    }
    if (haveReSTIR) {
        OPTIX_CHECK(optixSbtRecordPackHeader(m_pgRaygenReSTIR, &raygenReSTIRRec));
    }
    if (haveReSTIRVis) {
        OPTIX_CHECK(optixSbtRecordPackHeader(m_pgRaygenReSTIRVis, &raygenReSTIRVisRec));
    }
    if (haveReSTIRGI) {
        OPTIX_CHECK(optixSbtRecordPackHeader(m_pgRaygenReSTIRGI, &raygenReSTIRGIRec));
    }
    if (haveReSTIRPT) {
        OPTIX_CHECK(optixSbtRecordPackHeader(m_pgRaygenReSTIRPT, &raygenReSTIRPTRec));
    }
    OPTIX_CHECK(optixSbtRecordPackHeader(m_pgMissRadiance, &missR));
    OPTIX_CHECK(optixSbtRecordPackHeader(m_pgMissShadow,   &missS));
    OPTIX_CHECK(optixSbtRecordPackHeader(m_pgHitRadiance,  &hitR));
    OPTIX_CHECK(optixSbtRecordPackHeader(m_pgHitShadow,    &hitS));

    char* base = (char*)m_sbtRecordsBuf;
    size_t off = 0;
    CUDA_CHECK(cudaMemcpy(base + off, &raygenRec, recSize, cudaMemcpyHostToDevice));
    m_dRaygenRecord = (CUdeviceptr)(base + off);
    off += recSize;
    if (haveSplit) {
        CUDA_CHECK(cudaMemcpy(base + off, &raygenSplitRec, recSize, cudaMemcpyHostToDevice));
        m_dRaygenSplitRecord = (CUdeviceptr)(base + off);
        off += recSize;
    } else {
        m_dRaygenSplitRecord = 0;
    }
    if (haveReSTIR) {
        CUDA_CHECK(cudaMemcpy(base + off, &raygenReSTIRRec, recSize, cudaMemcpyHostToDevice));
        m_dRaygenReSTIRRecord = (CUdeviceptr)(base + off);
        off += recSize;
    } else {
        m_dRaygenReSTIRRecord = 0;
    }
    if (haveReSTIRVis) {
        CUDA_CHECK(cudaMemcpy(base + off, &raygenReSTIRVisRec, recSize, cudaMemcpyHostToDevice));
        m_dRaygenReSTIRVisRecord = (CUdeviceptr)(base + off);
        off += recSize;
    } else {
        m_dRaygenReSTIRVisRecord = 0;
    }
    if (haveReSTIRGI) {
        CUDA_CHECK(cudaMemcpy(base + off, &raygenReSTIRGIRec, recSize, cudaMemcpyHostToDevice));
        m_dRaygenReSTIRGIRecord = (CUdeviceptr)(base + off);
        off += recSize;
    } else {
        m_dRaygenReSTIRGIRecord = 0;
    }
    if (haveReSTIRPT) {
        CUDA_CHECK(cudaMemcpy(base + off, &raygenReSTIRPTRec, recSize, cudaMemcpyHostToDevice));
        m_dRaygenReSTIRPTRecord = (CUdeviceptr)(base + off);
        off += recSize;
    } else {
        m_dRaygenReSTIRPTRecord = 0;
    }
    CUDA_CHECK(cudaMemcpy(base + off, &missR, recSize, cudaMemcpyHostToDevice)); off += recSize;
    CUDA_CHECK(cudaMemcpy(base + off, &missS, recSize, cudaMemcpyHostToDevice)); off += recSize;
    const CUdeviceptr hitBase = (CUdeviceptr)(base + off);
    CUDA_CHECK(cudaMemcpy(base + off, &hitR,  recSize, cudaMemcpyHostToDevice)); off += recSize;
    CUDA_CHECK(cudaMemcpy(base + off, &hitS,  recSize, cudaMemcpyHostToDevice)); off += recSize;

    std::memset(&m_sbt, 0, sizeof(m_sbt));
    m_sbt.raygenRecord                = m_dRaygenRecord;  // default = regular raygen
    m_sbt.missRecordBase              = (CUdeviceptr)(base + recSize * numRaygens);
    m_sbt.missRecordStrideInBytes     = (unsigned int)recSize;
    m_sbt.missRecordCount             = 2;
    m_sbt.hitgroupRecordBase          = hitBase;
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

    // Use the regular raygen (in case a previous call swapped to raygenSplit).
    m_sbt.raygenRecord = m_dRaygenRecord;

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

#ifdef PATHTRACER_NRD_DLSS_ENABLED
void OptiXBackend::launchPathTraceSplit(
    const DeviceSceneData& scene,
    const CameraParams& camera,
    SplitSurfaceOutputs surfaces,
    uint32_t width, uint32_t height,
    uint32_t sampleIndex,
    bool enableEnvironment,
    uint32_t maxBounces,
    uint32_t samplesPerPixel)
{
    if (!m_initialized || !m_gasHandle || !m_dRaygenSplitRecord) return;

    LaunchParams lp;
    std::memset(&lp, 0, sizeof(lp));
    lp.scene   = scene;
    lp.camera  = camera;
    // accum/output/aux/gbuffer all unused by the split raygen — leave zeroed.
    lp.splitDiffuseRadianceHitDist  = surfaces.diffuseRadianceHitDist;
    lp.splitSpecularRadianceHitDist = surfaces.specularRadianceHitDist;
    lp.splitNormalRoughness         = surfaces.normalRoughness;
    lp.splitViewZ                   = surfaces.viewZ;
    lp.splitMotionVectors           = surfaces.motionVectors;
    lp.splitAlbedo                  = surfaces.albedo;
    lp.splitEmissive                = surfaces.emissive;
    lp.splitNdcDepth                = surfaces.ndcDepth;
    lp.width       = width;
    lp.height      = height;
    lp.sampleIndex = sampleIndex;
    lp.maxBounces  = maxBounces;
    lp.spp         = samplesPerPixel < 1 ? 1 : samplesPerPixel;
    lp.enableEnvironment = enableEnvironment ? 1u : 0u;
    lp.handle      = m_gasHandle;

    // Swap to the split raygen for this launch.
    m_sbt.raygenRecord = m_dRaygenSplitRecord;

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
#endif

bool OptiXBackend::launchReSTIRInitCandidatesOptiX(
    const DeviceSceneData& scene,
    const CameraParams&    camera,
    void*                  d_reservoirsCurr,
    void*                  d_surfacesCurr,
    uint32_t               width,
    uint32_t               height,
    uint32_t               sampleIndex,
    uint32_t               numCandidates)
{
    if (!m_initialized || !m_gasHandle || !m_dRaygenReSTIRRecord) return false;
    if (!d_reservoirsCurr || !d_surfacesCurr) return false;
    if (!scene.d_lightBVHNodes || scene.areaLightCount == 0) return false;

    LaunchParams lp;
    std::memset(&lp, 0, sizeof(lp));
    lp.scene  = scene;
    lp.camera = camera;
    // accum/output/aux/gbuffer and all split-output surfaces are unused by
    // the ReSTIR raygen — the memset above zeroes them. Only the scene +
    // camera + ReSTIR-specific fields need to be valid.
    lp.width       = width;
    lp.height      = height;
    lp.sampleIndex = sampleIndex;
    lp.maxBounces  = 1;   // unused here; set to 1 for defensiveness.
    lp.spp         = 1;
    lp.enableEnvironment = 0;
    lp.handle      = m_gasHandle;
    lp.restirReservoirsCurr = static_cast<ReSTIRReservoir*>(d_reservoirsCurr);
    lp.restirSurfacesCurr   = static_cast<ReSTIRSurface*>(d_surfacesCurr);
    lp.restirNumCandidates  = numCandidates;

    m_sbt.raygenRecord = m_dRaygenReSTIRRecord;

    CUDA_CHECK(cudaMemcpyAsync(
        (void*)m_dLaunchParams, &lp, sizeof(lp),
        cudaMemcpyHostToDevice, m_stream));

    OPTIX_CHECK_VOID(optixLaunch(
        m_pipeline, m_stream,
        m_dLaunchParams, sizeof(LaunchParams),
        &m_sbt,
        width, height, 1));
    CUDA_CHECK(cudaStreamSynchronize(m_stream));

    // Restore regular raygen so a subsequent launchPathTrace doesn't
    // accidentally fire the ReSTIR raygen.
    m_sbt.raygenRecord = m_dRaygenRecord;
    return true;
}

bool OptiXBackend::launchReSTIRVisibilityReuseOptiX(
    const DeviceSceneData& scene,
    void*                  d_reservoirsCurr,
    const void*            d_surfacesCurr,
    uint32_t               width,
    uint32_t               height)
{
    if (!m_initialized || !m_gasHandle || !m_dRaygenReSTIRVisRecord) return false;
    if (!d_reservoirsCurr || !d_surfacesCurr) return false;
    if (!scene.d_areaLights || scene.areaLightCount == 0) return false;

    LaunchParams lp;
    std::memset(&lp, 0, sizeof(lp));
    lp.scene  = scene;
    // camera unused by visibility raygen; memset zeroed it.
    lp.width       = width;
    lp.height      = height;
    lp.sampleIndex = 0;     // unused
    lp.maxBounces  = 1;
    lp.spp         = 1;
    lp.enableEnvironment = 0;
    lp.handle      = m_gasHandle;
    lp.restirReservoirsCurr = static_cast<ReSTIRReservoir*>(d_reservoirsCurr);
    lp.restirSurfacesCurr   = const_cast<ReSTIRSurface*>(
        static_cast<const ReSTIRSurface*>(d_surfacesCurr));
    lp.restirNumCandidates  = 0;  // unused

    m_sbt.raygenRecord = m_dRaygenReSTIRVisRecord;

    CUDA_CHECK(cudaMemcpyAsync(
        (void*)m_dLaunchParams, &lp, sizeof(lp),
        cudaMemcpyHostToDevice, m_stream));

    OPTIX_CHECK_VOID(optixLaunch(
        m_pipeline, m_stream,
        m_dLaunchParams, sizeof(LaunchParams),
        &m_sbt,
        width, height, 1));
    CUDA_CHECK(cudaStreamSynchronize(m_stream));

    // Restore regular raygen.
    m_sbt.raygenRecord = m_dRaygenRecord;
    return true;
}

bool OptiXBackend::launchReSTIRGIInitCandidatesOptiX(
    const DeviceSceneData& scene,
    const CameraParams&    camera,
    void*                  d_giReservoirsCurr,
    void*                  d_giSurfacesCurr,
    uint32_t               width,
    uint32_t               height,
    uint32_t               sampleIndex,
    bool                   enableEnvironment,
    uint32_t               numCandidates)
{
    if (!m_initialized || !m_gasHandle || !m_dRaygenReSTIRGIRecord) return false;
    if (!d_giReservoirsCurr || !d_giSurfacesCurr) return false;
    // GI doesn't strictly need an area-light list (env hits are valid), but
    // if there are none AND no env, the indirect Lo will always be zero and
    // running the pass is wasted work. Skip in that case.
    if ((!scene.d_areaLights || scene.areaLightCount == 0) && !enableEnvironment)
        return false;
    if (numCandidates < 1) numCandidates = 1;

    LaunchParams lp;
    std::memset(&lp, 0, sizeof(lp));
    lp.scene  = scene;
    lp.camera = camera;
    lp.width       = width;
    lp.height      = height;
    lp.sampleIndex = sampleIndex;
    lp.maxBounces  = 1;   // unused; GI raygen does its own bounce
    lp.spp         = 1;
    lp.enableEnvironment = enableEnvironment ? 1u : 0u;
    lp.handle      = m_gasHandle;
    lp.giReservoirsCurr   = static_cast<GIReservoir*>(d_giReservoirsCurr);
    lp.giSurfacesCurr     = static_cast<ReSTIRSurface*>(d_giSurfacesCurr);
    lp.giEnableEnvironment = enableEnvironment ? 1u : 0u;
    lp.giNumCandidates    = numCandidates;

    m_sbt.raygenRecord = m_dRaygenReSTIRGIRecord;

    CUDA_CHECK(cudaMemcpyAsync(
        (void*)m_dLaunchParams, &lp, sizeof(lp),
        cudaMemcpyHostToDevice, m_stream));

    OPTIX_CHECK_VOID(optixLaunch(
        m_pipeline, m_stream,
        m_dLaunchParams, sizeof(LaunchParams),
        &m_sbt,
        width, height, 1));
    CUDA_CHECK(cudaStreamSynchronize(m_stream));

    // Restore regular raygen.
    m_sbt.raygenRecord = m_dRaygenRecord;
    return true;
}

bool OptiXBackend::launchReSTIRPTInitCandidatesOptiX(
    const DeviceSceneData& scene,
    const CameraParams&    camera,
    void*                  d_ptReservoirsCurr,
    void*                  d_ptSurfacesCurr,
    uint32_t               width,
    uint32_t               height,
    uint32_t               sampleIndex,
    bool                   enableEnvironment,
    uint32_t               pathLength,
    uint32_t               numCandidates)
{
    if (!m_initialized || !m_gasHandle || !m_dRaygenReSTIRPTRecord) return false;
    if (!d_ptReservoirsCurr || !d_ptSurfacesCurr) return false;
    // Path-tracer postfix needs SOME light source. Skip when nothing in the
    // scene can contribute.
    if ((!scene.d_areaLights || scene.areaLightCount == 0) && !enableEnvironment)
        return false;
    if (numCandidates < 1) numCandidates = 1;

    LaunchParams lp;
    std::memset(&lp, 0, sizeof(lp));
    lp.scene  = scene;
    lp.camera = camera;
    lp.width       = width;
    lp.height      = height;
    lp.sampleIndex = sampleIndex;
    lp.maxBounces  = pathLength;
    lp.spp         = 1;
    lp.enableEnvironment   = enableEnvironment ? 1u : 0u;
    lp.handle              = m_gasHandle;
    lp.ptReservoirsCurr    = static_cast<GIReservoir*>(d_ptReservoirsCurr);
    lp.ptSurfacesCurr      = static_cast<ReSTIRSurface*>(d_ptSurfacesCurr);
    lp.ptPathLength        = pathLength;
    lp.ptNumCandidates     = numCandidates;
    // The PT raygen reads `giEnableEnvironment` for the env toggle (sharing
    // the same field name keeps OptiXPrograms cleaner — no extra param).
    lp.giEnableEnvironment = enableEnvironment ? 1u : 0u;

    m_sbt.raygenRecord = m_dRaygenReSTIRPTRecord;

    CUDA_CHECK(cudaMemcpyAsync(
        (void*)m_dLaunchParams, &lp, sizeof(lp),
        cudaMemcpyHostToDevice, m_stream));

    OPTIX_CHECK_VOID(optixLaunch(
        m_pipeline, m_stream,
        m_dLaunchParams, sizeof(LaunchParams),
        &m_sbt,
        width, height, 1));
    CUDA_CHECK(cudaStreamSynchronize(m_stream));

    m_sbt.raygenRecord = m_dRaygenRecord;
    return true;
}

void OptiXBackend::traceOcclusionRays(
    const float3* /*d_origins*/,
    const float3* /*d_targets*/,
    bool* /*d_visible*/,
    uint32_t /*rayCount*/)
{
    // BDPT stub — unused currently.
}
