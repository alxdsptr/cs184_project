cmake_minimum_required(VERSION 3.22)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
project(cuda_pathtracer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 75 86 89)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ── NRD + DLSS master switch ─────────────────────────────────
# Default OFF: binary must stay byte-identical to the pre-feature baseline.
option(PATHTRACER_ENABLE_NRD_DLSS
    "Enable NRD denoiser + DLSS-SR super-resolution (requires external/NRD submodule and DLSS SDK)"
    OFF)

# Location of the prebuilt DLSS SDK (only consulted when the option is ON).
set(PATHTRACER_DLSS_SDK_DIR "D:/examples/DLSS" CACHE PATH
    "Path to the NVIDIA DLSS (NGX) SDK root (contains include/ and lib/Windows_x86_64/)")

# ── OptiX master switch ──────────────────────────────────────
option(PATHTRACER_ENABLE_OPTIX
    "Enable OptiX ray tracing backend (requires OptiX 7+ SDK)"
    OFF)

set(PATHTRACER_OPTIX_SDK_DIR "D:/tools/Optix" CACHE PATH
    "Path to the NVIDIA OptiX 7+ SDK root (contains include/)")

# ── Dependencies ──────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)

# Vulkan SDK — honor VULKAN_SDK env var, fall back to machine-local install.
if(NOT DEFINED ENV{VULKAN_SDK} AND EXISTS "d:/tools/Vulkan")
    set(ENV{VULKAN_SDK} "d:/tools/Vulkan")
endif()
find_package(Vulkan REQUIRED)
if(NOT Vulkan_GLSLC_EXECUTABLE)
    # glslc usually ships alongside vulkan-1.lib; Vulkan::Vulkan won't expose
    # it on older CMake's FindVulkan, so look up manually.
    find_program(Vulkan_GLSLC_EXECUTABLE NAMES glslc
        HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin"
        REQUIRED)
endif()
message(STATUS "glslc: ${Vulkan_GLSLC_EXECUTABLE}")

include(cmake/CompilerFlags.cmake)
include(cmake/FetchGLFW.cmake)
include(cmake/FetchImGui.cmake)
include(cmake/FetchGLI.cmake)

# Assimp (existing submodule)
set(ASSIMP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_ASSIMP_TOOLS OFF CACHE BOOL "" FORCE)
set(ASSIMP_INSTALL OFF CACHE BOOL "" FORCE)
set(ASSIMP_NO_EXPORT ON CACHE BOOL "" FORCE)
set(ASSIMP_WARNINGS_AS_ERRORS OFF CACHE BOOL "" FORCE)
add_subdirectory(assimp)

# ── NRD / NRI / DLSS (conditional) ───────────────────────────
if(PATHTRACER_ENABLE_NRD_DLSS)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/NRD/CMakeLists.txt")
        message(FATAL_ERROR
            "PATHTRACER_ENABLE_NRD_DLSS=ON but external/NRD submodule is missing. "
            "Run: git submodule update --init --recursive")
    endif()

    # NRD's own CMake will FetchContent NRI + MathLib + ShaderMake when NRD_NRI=ON.
    set(NRD_NRI ON CACHE BOOL "" FORCE)
    set(NRD_EMBEDS_SPIRV_SHADERS ON CACHE BOOL "" FORCE)
    # We don't need D3D shaders for a Vulkan-only pathtracer.
    set(NRD_EMBEDS_DXIL_SHADERS OFF CACHE BOOL "" FORCE)
    set(NRD_EMBEDS_DXBC_SHADERS OFF CACHE BOOL "" FORCE)
    # Match our aux-buffer format (VK_FORMAT_R8G8B8A8_UNORM). Encoding 0 stores
    # normal as plain XYZ in RGB + linear roughness in A, which is what the
    # CUDA path-trace kernel writes. The NRD default (2 = R10G10B10A2_UNORM
    # with 10-bit oct packing) would make NRD decode garbage from an 8-bit
    # image and silently destroy denoising quality.
    set(NRD_NORMAL_ENCODING "0" CACHE STRING "" FORCE)
    set(NRD_ROUGHNESS_ENCODING "1" CACHE STRING "" FORCE)
    add_subdirectory(external/NRD)

    # DLSS (NGX) — prebuilt SDK, headers + import lib + runtime DLL.
    # Import libs come in two flavors:
    #   Release : nvsdk_ngx_d.lib     (MD  + ITERATOR_DEBUG_LEVEL=0)
    #   Debug   : nvsdk_ngx_d_dbg.lib (MDd + ITERATOR_DEBUG_LEVEL=2)
    # Picking the wrong one produces LNK2038 mismatches on MSVC.
    find_path(DLSS_INCLUDE_DIR
        NAMES nvsdk_ngx.h nvsdk_ngx_vk.h
        PATHS "${PATHTRACER_DLSS_SDK_DIR}/include"
        NO_DEFAULT_PATH
        REQUIRED)
    find_library(DLSS_LIBRARY_RELEASE
        NAMES nvsdk_ngx_d
        PATHS "${PATHTRACER_DLSS_SDK_DIR}/lib/Windows_x86_64/x64"
        NO_DEFAULT_PATH
        REQUIRED)
    find_library(DLSS_LIBRARY_DEBUG
        NAMES nvsdk_ngx_d_dbg
        PATHS "${PATHTRACER_DLSS_SDK_DIR}/lib/Windows_x86_64/x64"
        NO_DEFAULT_PATH)
    if(NOT DLSS_LIBRARY_DEBUG)
        # Fall back to release lib if debug variant is missing — will break Debug
        # builds but keeps Release functional.
        set(DLSS_LIBRARY_DEBUG "${DLSS_LIBRARY_RELEASE}")
    endif()
    # Generator-expression-friendly path for per-config linkage.
    set(DLSS_LIBRARY "$<IF:$<CONFIG:Debug>,${DLSS_LIBRARY_DEBUG},${DLSS_LIBRARY_RELEASE}>")

    set(DLSS_RUNTIME_DLL_RELEASE "${PATHTRACER_DLSS_SDK_DIR}/lib/Windows_x86_64/rel/nvngx_dlss.dll")
    set(DLSS_RUNTIME_DLL_DEBUG   "${PATHTRACER_DLSS_SDK_DIR}/lib/Windows_x86_64/dev/nvngx_dlss.dll")
    if(NOT EXISTS "${DLSS_RUNTIME_DLL_RELEASE}")
        message(FATAL_ERROR "DLSS runtime DLL not found at ${DLSS_RUNTIME_DLL_RELEASE}")
    endif()
    if(NOT EXISTS "${DLSS_RUNTIME_DLL_DEBUG}")
        # Fall back to release DLL for Debug — NGX will still function, the
        # only loss is debug symbol visibility.
        set(DLSS_RUNTIME_DLL_DEBUG "${DLSS_RUNTIME_DLL_RELEASE}")
    endif()
    set(DLSS_RUNTIME_DLL "$<IF:$<CONFIG:Debug>,${DLSS_RUNTIME_DLL_DEBUG},${DLSS_RUNTIME_DLL_RELEASE}>")

    message(STATUS "DLSS include:   ${DLSS_INCLUDE_DIR}")
    message(STATUS "DLSS lib (rel): ${DLSS_LIBRARY_RELEASE}")
    message(STATUS "DLSS lib (dbg): ${DLSS_LIBRARY_DEBUG}")
    message(STATUS "DLSS dll (rel): ${DLSS_RUNTIME_DLL_RELEASE}")
    message(STATUS "DLSS dll (dbg): ${DLSS_RUNTIME_DLL_DEBUG}")
endif()

# ── SPIR-V shader compilation ────────────────────────────────
set(SHADER_SRC_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
set(SHADER_OUT_DIR  ${CMAKE_CURRENT_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SHADER_OUT_DIR})

set(SHADER_FILES
    fullscreen_quad_vk.vert
    fullscreen_quad_vk.frag
)
if(PATHTRACER_ENABLE_NRD_DLSS)
    list(APPEND SHADER_FILES
        composite_tonemap.frag
        composite_linear.frag
        tonemap_only.frag
    )
endif()
set(SPV_OUTPUTS)
foreach(SHADER IN LISTS SHADER_FILES)
    set(IN  ${SHADER_SRC_DIR}/${SHADER})
    set(OUT ${SHADER_OUT_DIR}/${SHADER}.spv)
    add_custom_command(
        OUTPUT  ${OUT}
        COMMAND ${Vulkan_GLSLC_EXECUTABLE} ${IN} -o ${OUT}
        DEPENDS ${IN}
        VERBATIM)
    list(APPEND SPV_OUTPUTS ${OUT})
endforeach()
add_custom_target(shaders DEPENDS ${SPV_OUTPUTS})

# ── Main executable ──────────────────────────────────────────
add_executable(pathtracer
    # App
    src/main.cpp
    src/app/Application.cpp
    # Core
    src/core/Math.cpp
    src/core/Camera.cpp
    src/core/SceneCollider.cpp
    # Scene
    src/scene/Scene.cpp
    src/scene/SceneLoader.cpp
    src/scene/PbrtLoader.cpp
    src/scene/Texture.cpp
    src/scene/utils.cpp
    # Display
    src/display/VulkanDisplay.cpp
    # GUI
    src/gui/GUI.cpp
    # Render
    src/render/Renderer.cpp
    # Accel (CPU-only BVH construction)
    src/accel/SAH_BVH.cpp
    src/accel/LightBVH.cpp
)

# CUDA sources
target_sources(pathtracer PRIVATE
    src/gpu/DeviceScene.cu
    src/backend/CUDABackend.cu
    src/render/PathTraceKernel.cu
    src/render/Tonemapping.cu
    src/render/AccumulationBuffer.cu
    src/render/AuxBuffers.cu
    src/render/ReSTIR.cu
)

add_dependencies(pathtracer shaders)

target_include_directories(pathtracer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/assimp/include
    ${CMAKE_BINARY_DIR}/assimp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/assimp/contrib/stb
    ${CMAKE_BINARY_DIR}/_deps/glfw-src/deps
)

target_link_libraries(pathtracer PRIVATE
    Vulkan::Vulkan
    glfw
    imgui
    assimp
    gli_headers
    CUDA::cudart
)

# ── Conditional OptiX sources / build rules ──────────────────
if(PATHTRACER_ENABLE_OPTIX)
    if(NOT EXISTS "${PATHTRACER_OPTIX_SDK_DIR}/include/optix.h")
        message(FATAL_ERROR
            "PATHTRACER_ENABLE_OPTIX=ON but ${PATHTRACER_OPTIX_SDK_DIR}/include/optix.h is missing. "
            "Install the OptiX SDK or set -DPATHTRACER_OPTIX_SDK_DIR=<path>.")
    endif()

    target_sources(pathtracer PRIVATE
        src/backend/OptiXBackend.cpp
        src/backend/OptiXBackend.h
        src/backend/OptiXLaunchParams.h
        src/backend/OptiXSBTRecords.h
    )
    target_compile_definitions(pathtracer PRIVATE PATHTRACER_OPTIX_ENABLED=1)
    target_include_directories(pathtracer PRIVATE
        ${PATHTRACER_OPTIX_SDK_DIR}/include
    )
    # Driver API needed for CUdeviceptr + CUstream (OptiX types).
    target_link_libraries(pathtracer PRIVATE CUDA::cuda_driver)

    # Compile OptiX device programs to OptiX-IR.
    set(OPTIX_PROGRAMS_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/backend/OptiXPrograms.cu)
    set(OPTIX_PROGRAMS_IR  ${CMAKE_CURRENT_BINARY_DIR}/optix_programs.optixir)
    # Headers that change LaunchParams / DeviceSceneData layout MUST trigger a
    # rebuild — otherwise the shipped .optixir bakes in a stale
    # pipelineLaunchParamsSize and runtime barfs with OPTIX_ERROR_INVALID_VALUE
    # (7001) "launch param size configured by pipeline link options ... is
    # smaller than pipelineParamsSize". nvcc --optix-ir doesn't emit a depfile,
    # so we list the big ones by hand.
    set(OPTIX_PROGRAMS_DEPS
        ${CMAKE_CURRENT_SOURCE_DIR}/src/backend/OptiXLaunchParams.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/backend/OptiXSBTRecords.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/DeviceScene.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/MaterialGPU.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/LightGPU.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/AreaLightGPU.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/Sampling.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/core/Camera.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/render/AuxBuffers.h
    )
    add_custom_command(
        OUTPUT  ${OPTIX_PROGRAMS_IR}
        COMMAND ${CMAKE_CUDA_COMPILER}
                --optix-ir
                -arch=compute_75
                -rdc=true
                -std=c++17
                --use_fast_math
                --expt-relaxed-constexpr
                -I${CMAKE_CURRENT_SOURCE_DIR}/src
                -I${PATHTRACER_OPTIX_SDK_DIR}/include
                "-I$<JOIN:${CUDAToolkit_INCLUDE_DIRS},;-I>"
                -o ${OPTIX_PROGRAMS_IR}
                ${OPTIX_PROGRAMS_SRC}
        DEPENDS ${OPTIX_PROGRAMS_SRC} ${OPTIX_PROGRAMS_DEPS}
        COMMENT "Compiling OptiX device programs to OptiX-IR"
        VERBATIM)
    add_custom_target(optix_programs_ir DEPENDS ${OPTIX_PROGRAMS_IR})
    add_dependencies(pathtracer optix_programs_ir)

    # Ship the .optixir next to the exe so the runtime can find it.
    add_custom_command(TARGET pathtracer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${OPTIX_PROGRAMS_IR}
            $<TARGET_FILE_DIR:pathtracer>/optix_programs.optixir
        VERBATIM)
endif()

# ── Conditional NRD + DLSS sources / links ───────────────────
if(PATHTRACER_ENABLE_NRD_DLSS)
    target_sources(pathtracer PRIVATE
        src/interop/VulkanImageInterop.cpp
        src/render/VulkanSharedAuxBuffers.cpp
        src/render/PathTraceKernelSplit.cu
        src/postfx/NRDContext.cpp
        src/postfx/CompositePass.cpp
        src/postfx/DLSSContext.cpp
    )
    target_compile_definitions(pathtracer PRIVATE PATHTRACER_NRD_DLSS_ENABLED=1)
    target_include_directories(pathtracer PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/external/NRD/Include
        ${CMAKE_CURRENT_SOURCE_DIR}/external/NRD/Integration
        ${DLSS_INCLUDE_DIR}
    )
    # NRD's FetchContent pulls NRI into the build tree; it exposes a NRI CMake target.
    target_link_libraries(pathtracer PRIVATE
        NRD
        NRI
        "${DLSS_LIBRARY}"
    )
endif()

set_target_properties(pathtracer PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_compile_options(pathtracer PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --use_fast_math
        -lineinfo
    >
)

# Copy SPIR-V alongside the executable so runtime relative lookup works,
# regardless of whether the working directory is build/ or its subdir.
add_custom_command(TARGET pathtracer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${SHADER_OUT_DIR}
        $<TARGET_FILE_DIR:pathtracer>/shaders
    VERBATIM)

# Ship the NGX runtime DLL next to the executable when DLSS is enabled.
# DLSS_RUNTIME_DLL is a generator expression that resolves per-config.
if(PATHTRACER_ENABLE_NRD_DLSS)
    add_custom_command(TARGET pathtracer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${DLSS_RUNTIME_DLL}"
            $<TARGET_FILE_DIR:pathtracer>/
        VERBATIM)

    # NRD is built as a shared library by default (NRD_STATIC_LIBRARY=OFF),
    # so ship NRD/NRI runtime DLLs alongside the executable.
    if(NOT NRD_STATIC_LIBRARY)
        add_custom_command(TARGET pathtracer POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:NRD>"
                $<TARGET_FILE_DIR:pathtracer>/
            VERBATIM)

        add_custom_command(TARGET pathtracer POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:NRI>"
                $<TARGET_FILE_DIR:pathtracer>/
            VERBATIM)
    endif()
endif()
