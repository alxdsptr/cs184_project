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
    add_subdirectory(external/NRD)

    # DLSS (NGX) — prebuilt SDK, headers + import lib + runtime DLL
    find_path(DLSS_INCLUDE_DIR
        NAMES nvsdk_ngx.h nvsdk_ngx_vk.h
        PATHS "${PATHTRACER_DLSS_SDK_DIR}/include"
        NO_DEFAULT_PATH
        REQUIRED)
    find_library(DLSS_LIBRARY
        NAMES nvsdk_ngx_d
        PATHS "${PATHTRACER_DLSS_SDK_DIR}/lib/Windows_x86_64/x64"
        NO_DEFAULT_PATH
        REQUIRED)
    set(DLSS_RUNTIME_DLL "${PATHTRACER_DLSS_SDK_DIR}/lib/Windows_x86_64/rel/nvngx_dlss.dll"
        CACHE FILEPATH "")
    if(NOT EXISTS "${DLSS_RUNTIME_DLL}")
        message(FATAL_ERROR "DLSS runtime DLL not found at ${DLSS_RUNTIME_DLL}")
    endif()
    message(STATUS "DLSS include: ${DLSS_INCLUDE_DIR}")
    message(STATUS "DLSS lib:     ${DLSS_LIBRARY}")
    message(STATUS "DLSS dll:     ${DLSS_RUNTIME_DLL}")
endif()

# ── SPIR-V shader compilation ────────────────────────────────
set(SHADER_SRC_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
set(SHADER_OUT_DIR  ${CMAKE_CURRENT_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SHADER_OUT_DIR})

set(SHADER_FILES
    fullscreen_quad_vk.vert
    fullscreen_quad_vk.frag
)
# NRD+DLSS shaders (composite_tonemap.frag, composite_linear.frag,
# tonemap_only.frag) are appended here by Step 6 once they exist.
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
)

# CUDA sources
target_sources(pathtracer PRIVATE
    src/gpu/DeviceScene.cu
    src/backend/CUDABackend.cu
    src/render/PathTraceKernel.cu
    src/render/Tonemapping.cu
    src/render/AccumulationBuffer.cu
    src/render/AuxBuffers.cu
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

# ── Conditional NRD + DLSS sources / links ───────────────────
if(PATHTRACER_ENABLE_NRD_DLSS)
    target_sources(pathtracer PRIVATE
        src/interop/VulkanImageInterop.cpp
        src/render/VulkanSharedAuxBuffers.cpp
        src/render/PathTraceKernelSplit.cu
        # Added as they come online (NRDContext.cpp, DLSSContext.cpp, CompositePass.cpp)
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
if(PATHTRACER_ENABLE_NRD_DLSS AND EXISTS "${DLSS_RUNTIME_DLL}")
    add_custom_command(TARGET pathtracer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${DLSS_RUNTIME_DLL}"
            $<TARGET_FILE_DIR:pathtracer>/
        VERBATIM)
endif()
