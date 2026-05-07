include(FetchContent)
FetchContent_Declare(
    fpng
    GIT_REPOSITORY https://github.com/richgel999/fpng.git
    GIT_TAG        v1.0.6
)
FetchContent_MakeAvailable(fpng)

# fpng has no CMakeLists.txt at the repo root — wrap the two source files in a
# static library target ourselves, mirroring how FetchImGui.cmake handles imgui.
add_library(fpng STATIC
    ${fpng_SOURCE_DIR}/src/fpng.cpp
)
target_include_directories(fpng PUBLIC
    ${fpng_SOURCE_DIR}/src
)

# fpng's fast path uses SSE 4.1 + PCLMUL intrinsics. On MSVC these are always
# available with no flag. On GCC/Clang we need to enable them explicitly,
# otherwise the compiler errors out on the intrinsic include.
if(NOT MSVC)
    target_compile_options(fpng PRIVATE -msse4.1 -mpclmul)
endif()
