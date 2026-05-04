#pragma once
#include <cstdio>
#include <cstdlib>

#define LOG_INFO(fmt, ...)  fprintf(stdout, "[INFO]  " fmt "\n", ##__VA_ARGS__)
#define LOG_WARN(fmt, ...)  fprintf(stderr, "[WARN]  " fmt "\n", ##__VA_ARGS__)
#define LOG_ERROR(fmt, ...) fprintf(stderr, "[ERROR] " fmt "\n", ##__VA_ARGS__)

// LOG_DEBUG is silenced by default. Enable via:
//   PATHTRACER_VERBOSE=1   environment variable (any non-empty value)
// Use this for high-frequency / per-resource logs (per-material diagnostics,
// per-texture decode info, BVH build stats) that are useful for debugging
// asset-load issues but produce hundreds of lines on real scenes.
//
// Cost: one getenv at first call per call site, then a hot global bool.
inline bool g_pathtracerLogDebugEnabled() {
    static const bool kEnabled = []() {
        const char* v = std::getenv("PATHTRACER_VERBOSE");
        return v != nullptr && v[0] != '\0' && !(v[0] == '0' && v[1] == '\0');
    }();
    return kEnabled;
}
#define LOG_DEBUG(fmt, ...) do { \
    if (g_pathtracerLogDebugEnabled()) { \
        fprintf(stdout, "[DEBUG] " fmt "\n", ##__VA_ARGS__); \
    } \
} while (0)
