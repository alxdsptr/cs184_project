#pragma once
#include "scene/Scene.h"
#include <cstdint>
#include <string>

// Binary sidecar cache for the post-Assimp `Scene` produced by SceneLoader.
// A cache hit skips the entire Assimp parse + post-process pipeline (which is
// the dominant cost on heavy scenes like MEASURE_SEVEN / Bistro), turning a
// scene reload into a single linear file read.
//
// File layout: see SceneCache.cpp. The cache is keyed on
//   - the source file's mtime + size
//   - the loader options that affect output (sgMode, emissive target lum)
//   - a struct-layout fingerprint computed at compile time (sizeof of every
//     scene-data struct combined). Any header-level field addition flips this
//     and silently invalidates every existing cache.
//   - a manual cache version constant (bump when loader semantics change in a
//     way the fingerprint won't catch — e.g. tweaks to applyUnitScaling).
struct SceneCacheKey {
    int64_t  srcMtime          = 0;
    uint64_t srcSize           = 0;
    uint32_t sgMode            = 0;
    float    emissiveTargetLum = 0.0f;
};

class SceneCache {
public:
    // Returns true and populates `out` if a valid cache exists for (srcPath, key).
    // Returns false (without modifying `out` beyond default-construction) on any
    // mismatch or read error — caller falls back to the slow Assimp path.
    static bool tryLoad(const std::string& srcPath, const SceneCacheKey& key, Scene& out);

    // Writes `scene` to the cache for (srcPath, key). Returns true on success.
    // Best-effort: a write failure logs a warning but is non-fatal.
    static bool save(const std::string& srcPath, const SceneCacheKey& key, const Scene& scene);

    // Sidecar path: `<srcPath>.scenecache`.
    static std::string cachePathFor(const std::string& srcPath);

    // Build a SceneCacheKey from the source file's filesystem stat + the load
    // options. Returns false if the source file can't be stat'd.
    static bool buildKey(const std::string& srcPath,
                         uint32_t sgMode, float emissiveTargetLum,
                         SceneCacheKey& outKey);
};
