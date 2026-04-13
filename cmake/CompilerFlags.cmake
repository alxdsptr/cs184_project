if(MSVC)
    # Only apply MSVC flags to C/CXX, not CUDA
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/W3>)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/permissive->)
    string(REPLACE "/O2" "/Ox" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
endif()
