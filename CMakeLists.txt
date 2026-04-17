cmake_minimum_required(VERSION 3.22)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
project(cuda_pathtracer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 75 89)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

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

# ── SPIR-V shader compilation ────────────────────────────────
set(SHADER_SRC_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
set(SHADER_OUT_DIR  ${CMAKE_CURRENT_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SHADER_OUT_DIR})

set(SHADER_FILES
    fullscreen_quad_vk.vert
    fullscreen_quad_vk.frag
)
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
