cmake_minimum_required(VERSION 3.11)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
project(Assignment3)

#-------------------------------------------------------------------------------
# Build options
#-------------------------------------------------------------------------------
option(BUILD_DEBUG     "Build with debug settings"    OFF)
option(BUILD_DOCS      "Build documentation"          OFF)
option(BUILD_CUDA      "Build with CUDA GPU path tracing" OFF)


set(BUILD_DEBUG ${BUILD_DEBUG} CACHE BOOL "Build debug" FORCE)

if (BUILD_DEBUG)
  set(CMAKE_BUILD_TYPE Debug)
endif()

#-------------------------------------------------------------------------------
# Set target
#-------------------------------------------------------------------------------
set(APPLICATION_3_1_SOURCE

    # Scene Object & Structure
    src/scene/sphere.cpp
    src/scene/triangle.cpp
    src/scene/light.cpp
    src/scene/bvh.cpp
    src/scene/bbox.cpp

    # Pathtracer
    src/pathtracer/camera.cpp
    src/pathtracer/bsdf.cpp
    src/pathtracer/pathtracer.cpp

    # Imgui
    src/imgui/imgui.cpp
    src/imgui/imgui_widgets.cpp
    src/imgui/imgui_draw.cpp
    src/imgui/imgui_tables.cpp
    src/imgui/backends/imgui_impl_glfw.cpp
    src/imgui/backends/imgui_impl_opengl2.cpp

    src/application/pathtracer_launcher_gui.cpp
)

set(APPLICATION_3_2_SOURCE
    src/scene/object.cpp

    # Collada Parser
    src/scene/collada/collada.cpp
    src/scene/collada/camera_info.cpp
    src/scene/collada/light_info.cpp
    src/scene/collada/sphere_info.cpp
    src/scene/collada/polymesh_info.cpp
    src/scene/collada/material_info.cpp

    # Dynamic Scene
    src/scene/gl_scene/mesh.cpp
    src/scene/gl_scene/scene.cpp
    src/scene/gl_scene/sphere.cpp

    # MeshEdit
    src/util/halfEdgeMesh.cpp
    src/application/meshEdit.cpp

    src/pathtracer/sampler.cpp
    src/pathtracer/advanced_bsdf.cpp
    src/scene/environment_light.cpp
    src/pathtracer/camera_lens.cpp
    src/pathtracer/raytraced_renderer.cpp

    # misc
    src/util/sphere_drawing.cpp
    src/util/lodepng.cpp

    # Application
    src/application/application.cpp
    src/application/main.cpp
    src/application/visual_debugger.cpp

    src/imgui/imgui.cpp
    src/imgui/imgui_widgets.cpp
    src/imgui/imgui_draw.cpp
    src/imgui/imgui_tables.cpp
    src/imgui/backends/imgui_impl_glfw.cpp
    src/imgui/backends/imgui_impl_opengl2.cpp
)

set(APPLICATION_HEADERS
    # Collada Parser
    src/scene/collada/camera_info.h
    src/scene/collada/collada_info.h
    src/scene/collada/collada.h
    src/scene/collada/light_info.h
    src/scene/collada/material_info.h
    src/scene/collada/polymesh_info.h
    src/scene/collada/sphere_info.h
    # Dynamic Scene
    src/scene/gl_scene/ambient_light.h
    src/scene/gl_scene/area_light.h
    src/scene/gl_scene/directional_light.h
    src/scene/gl_scene/draw_style.h
    src/scene/gl_scene/environment_light.h
    src/scene/gl_scene/material.h
    src/scene/gl_scene/mesh_view.h
    src/scene/gl_scene/mesh.h
    src/scene/gl_scene/point_light.h
    src/scene/gl_scene/scene.h
    src/scene/gl_scene/sphere.h
    src/scene/gl_scene/spot_light.h
    # Scene Object & Structure
    src/scene/aggregate.h
    src/scene/bbox.h
    src/scene/bvh.h
    src/scene/environment_light.h
    src/scene/light.h
    src/scene/object.h
    src/scene/primitive.h
    src/scene/scene.h
    src/scene/sphere.h
    src/scene/triangle.h
    # MeshEdit
    src/util/halfEdgeMesh.h
    src/util/image.h
    src/util/mutablePriorityQueue.h
    src/util/random_util.h
    src/util/work_queue.h
    # Pathtracer
    src/pathtracer/bsdf.h
    src/pathtracer/camera.h
    src/pathtracer/intersection.h
    src/pathtracer/pathtracer.h
    src/pathtracer/ray.h
    src/pathtracer/raytraced_renderer.h
    src/pathtracer/sampler.h
    # misc
    src/util/sphere_drawing.h
    src/util/lodepng.h
    # Application
    src/application/application.h
    src/application/meshEdit.h
    src/application/renderer.h
)

if (WIN32)
    list(APPEND APPLICATION_3_1_SOURCE src/util/win32/getopt.c)
    if(MSVC)


      # Disable MSVC iterator debugging to avoid O(n^2) overhead when
      # destroying large containers of std::list iterators (e.g., in
      # HalfedgeMesh::build). Without this, debug builds hang on large meshes.
      add_compile_definitions(_ITERATOR_DEBUG_LEVEL=0)

    endif(MSVC)
endif()

if (WIN32 OR APPLE)
  add_library(pt31 STATIC ${APPLICATION_3_1_SOURCE} ${APPLICATION_HEADERS})
else()
  add_library(pt31 SHARED ${APPLICATION_3_1_SOURCE} ${APPLICATION_HEADERS})
endif()

target_include_directories(pt31 PUBLIC src)

add_executable(pathtracer ${APPLICATION_3_2_SOURCE} ${APPLICATION_HEADERS})
target_include_directories(pathtracer PUBLIC src)

target_link_libraries(pt31 PUBLIC CGL)
target_link_libraries(pathtracer PUBLIC pt31)

set(CGL_INCLUDE_DIRS CGL/include CGL/deps/glew/include CGL/deps/glfw/include ./src/imgui ./src/imgui/backends)

#-------------------------------------------------------------------------------
# Find dependencies
#-------------------------------------------------------------------------------
add_subdirectory(CGL)
target_include_directories(pt31 PUBLIC ${CGL_INCLUDE_DIRS})
target_include_directories(pathtracer PUBLIC ${CGL_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CGL_CXX_FLAGS}")

set(OpenGL_GL_PREFERENCE LEGACY)
find_package(OpenGL REQUIRED)
target_link_libraries(pt31 PUBLIC OpenGL::GL)
target_link_libraries(pt31 PUBLIC OpenGL::GLU)

#-------------------------------------------------------------------------------
# CUDA support
#-------------------------------------------------------------------------------
if(BUILD_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)

  set(CMAKE_CUDA_ARCHITECTURES "75;86" CACHE STRING "CUDA architectures")

  set(CUDA_SOURCES
    src/cuda/cuda_pathtracer.cu
  )

  set(CUDA_CPP_SOURCES
    src/cuda/cuda_scene.cpp
  )

  add_library(pt31_cuda STATIC ${CUDA_SOURCES})
  target_include_directories(pt31_cuda PUBLIC src)
  target_include_directories(pt31_cuda PUBLIC ${CGL_INCLUDE_DIRS})
  set_target_properties(pt31_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
  )
  if(MSVC)
    target_compile_options(pt31_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--std=c++17>)
  else()
    target_compile_options(pt31_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--std=c++14>)
  endif()

  # Add CUDA scene linearization code (compiled as regular C++)
  target_sources(pt31 PRIVATE ${CUDA_CPP_SOURCES})

  target_link_libraries(pt31 PUBLIC pt31_cuda CUDA::cudart)
  target_compile_definitions(pt31 PUBLIC USE_CUDA=1)
  target_compile_definitions(pt31_cuda PUBLIC USE_CUDA=1)
  target_compile_definitions(pathtracer PUBLIC USE_CUDA=1)
endif()

#-------------------------------------------------------------------------------
# Add subdirectories
#-------------------------------------------------------------------------------

# build documentation
if(BUILD_DOCS)
  find_package(DOXYGEN)
  if(DOXYGEN_FOUND AND BUILD_DOCS)
    add_subdirectory(docs)
  endif()
endif()

# Install settings
set(CMAKE_INSTALL_PREFIX "${Assignment1_SOURCE_DIR}/")
