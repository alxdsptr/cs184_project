#pragma once
#include <optix.h>

template <typename T>
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecord {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T    data;
};

struct EmptyData {};

using RaygenRecord = SbtRecord<EmptyData>;
using MissRecord   = SbtRecord<EmptyData>;
using HitRecord_   = SbtRecord<EmptyData>;
