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
find_package(OpenGL REQUIRED)

include(cmake/CompilerFlags.cmake)
include(cmake/FetchGLFW.cmake)
include(cmake/FetchGLEW.cmake)
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
    src/scene/Texture.cpp
    # Display
    src/display/GLDisplay.cpp
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

target_include_directories(pathtracer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/assimp/include
    ${CMAKE_BINARY_DIR}/assimp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/assimp/contrib/stb
    ${CMAKE_BINARY_DIR}/_deps/glfw-src/deps
)

target_link_libraries(pathtracer PRIVATE
    OpenGL::GL
    glfw
    libglew_static
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
