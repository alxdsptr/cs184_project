include(FetchContent)

FetchContent_Declare(
    gli
    GIT_REPOSITORY https://github.com/g-truc/gli.git
    GIT_TAG        0.8.2
)

FetchContent_GetProperties(gli)
if(NOT gli_POPULATED)
    FetchContent_Populate(gli)
endif()

# Use gli as header-only in this project to avoid building upstream tests.
add_library(gli_headers INTERFACE)
target_include_directories(gli_headers INTERFACE
    ${gli_SOURCE_DIR}
    ${gli_SOURCE_DIR}/external
)
