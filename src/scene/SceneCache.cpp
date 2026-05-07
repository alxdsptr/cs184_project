// TODO: replace serializers and compile-time cache version checksum with better methods like
// compile-time reflection via Boost.PBR or maybe even macros.

#include "scene/SceneCache.h"
#include "util/Log.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <type_traits>

// ── On-disk format ─────────────────────────────────────────────
//
//   [u32 magic = 'SCN1']
//   [u32 cacheVersion]            // bump on loader semantic changes
//   [u64 structFingerprint]       // sizeof()-based; auto-invalidates on field add
//   [SceneCacheKey raw]           // mtime + size + sgMode + targetLum
//   <scene payload>               // see serialize(v, scene) below
//   [u32 magic-end = 'SCNE']      // detects truncated writes
//
// All read/write logic is shared via a visitor pattern: each non-POD type
// has ONE `serialize(V&, T&)` that both Reader and Writer call into. POD
// vectors short-circuit to a single bulk fread/fwrite via `if constexpr`.

namespace {

constexpr uint32_t kCacheMagic    = 0x314E4353; // 'SCN1' (little-endian on disk)
constexpr uint32_t kCacheMagicEnd = 0x454E4353; // 'SCNE'
constexpr uint32_t kCacheVersion  = 1;

// Compile-time sizeof fingerprint. Any struct field addition / reordering /
// padding change flips this and silently invalidates every existing cache.
constexpr uint64_t kStructFingerprint =
    (uint64_t)sizeof(TriangleMesh)        *   3 +
    (uint64_t)sizeof(PBRMaterial)         *   5 +
    (uint64_t)sizeof(PointLight)          *   7 +
    (uint64_t)sizeof(DirectionalLight)    *  11 +
    (uint64_t)sizeof(TriangleAreaLight)   *  13 +
    (uint64_t)sizeof(SceneNode)           *  17 +
    (uint64_t)sizeof(MeshNodeBinding)     *  19 +
    (uint64_t)sizeof(AnimationClip)       *  23 +
    (uint64_t)sizeof(AnimChannelTrack)    *  29 +
    (uint64_t)sizeof(SceneCamera)         *  31 +
    (uint64_t)sizeof(AABB)                *  37 +
    (uint64_t)sizeof(float3)              *  41 +
    (uint64_t)sizeof(float4)              *  43 +
    (uint64_t)sizeof(float4x4)            *  47;

// Sanity caps on counts read from disk (prevents a corrupt header from
// driving a multi-GB allocation).
constexpr uint64_t kMaxVecElems   = 2'000'000'000ull;
constexpr uint64_t kMaxStringLen  =        16'777'216ull; // 16 MiB

// ── Visitors ──────────────────────────────────────────────────
// Each type has ONE serialize() function (below) that both visitors share.
// Writer takes T& (rather than const T&) so the same serialize() body works
// for both directions; Writer simply never mutates.

struct Writer {
    std::FILE* f;
    bool ok = true;

    void raw(const void* p, size_t n) {
        if (!ok) return;
        if (std::fwrite(p, 1, n, f) != n) ok = false;
    }
    template <typename T>
    std::enable_if_t<std::is_trivially_copyable_v<T>>
    process(T& v) { raw(&v, sizeof(T)); }

    void process(std::string& s) {
        uint32_t n = (uint32_t)s.size();
        process(n);
        if (n) raw(s.data(), n);
    }

    template <typename T>
    void process(std::vector<T>& v);
};

struct Reader {
    std::FILE* f;
    bool ok = true;

    bool raw(void* p, size_t n) {
        if (!ok) return false;
        if (std::fread(p, 1, n, f) != n) { ok = false; return false; }
        return true;
    }
    template <typename T>
    std::enable_if_t<std::is_trivially_copyable_v<T>>
    process(T& v) { raw(&v, sizeof(T)); }

    void process(std::string& s) {
        uint32_t n = 0;
        process(n);
        if (!ok || n > kMaxStringLen) { ok = false; return; }
        s.resize(n);
        if (n) raw(s.data(), n);
    }

    template <typename T>
    void process(std::vector<T>& v);
};

// Forward decls so vector::process can recurse into per-element serialize().
template <typename V> void serialize(V& v, TriangleMesh& m);
template <typename V> void serialize(V& v, PBRMaterial& m);
template <typename V> void serialize(V& v, SceneNode& n);
template <typename V> void serialize(V& v, AnimChannelTrack& t);
template <typename V> void serialize(V& v, AnimationClip& a);

// Vector dispatch — POD elements get bulk fread/fwrite, the rest recurse.
template <typename T>
void Writer::process(std::vector<T>& v) {
    uint64_t n = (uint64_t)v.size();
    process(n);
    if constexpr (std::is_trivially_copyable_v<T>) {
        if (n) raw(v.data(), (size_t)n * sizeof(T));
    } else {
        for (auto& e : v) serialize(*this, e);
    }
}
template <typename T>
void Reader::process(std::vector<T>& v) {
    uint64_t n = 0;
    process(n);
    if (!ok || n > kMaxVecElems) { ok = false; return; }
    v.resize((size_t)n);
    if constexpr (std::is_trivially_copyable_v<T>) {
        if (n) raw(v.data(), (size_t)n * sizeof(T));
    } else {
        for (auto& e : v) serialize(*this, e);
    }
}

// ── Per-type serializers ─────────────────────────────────────
// These are the ONLY places that name struct fields. The runtime-only
// CUDA texture handles in PBRMaterial / TriangleAreaLight are deliberately
// not included — they're 0 at save time (Application binds them after
// SceneLoader::load returns) and stay 0 after a cache hit, until Application
// re-binds them on the reload path.

template <typename V>
void serialize(V& v, TriangleMesh& m) {
    v.process(m.positions);
    v.process(m.normals);
    v.process(m.tangents);
    v.process(m.uvs);
    v.process(m.indices);
    v.process(m.materialIndex);
}

template <typename V>
void serialize(V& v, PBRMaterial& m) {
    v.process(m.albedo);
    v.process(m.roughness);
    v.process(m.metallic);
    v.process(m.emission);
    v.process(m.emissionStrength);
    v.process(m.ior);
    v.process(m.transmission);
    v.process(m.pureDiffuse);
    v.process(m.useSpecularGlossiness);
    v.process(m.specularColor);
    v.process(m.glossiness);
    v.process(m.specularGlossAlphaIsGlossiness);
    v.process(m.useFBXCustomPacking);
    v.process(m.useFBXUEPacking);
    v.process(m.albedoTexPath);
    v.process(m.normalTexPath);
    v.process(m.metallicRoughTexPath);
    v.process(m.emissiveTexPath);
    v.process(m.specularGlossTexPath);
}

template <typename V>
void serialize(V& v, SceneNode& n) {
    v.process(n.name);
    v.process(n.parent);
    v.process(n.localRest);
    v.process(n.worldRest);
    v.process(n.animChannel);
    v.process(n.animated);
    v.process(n.meshCount);
}

template <typename V>
void serialize(V& v, AnimChannelTrack& t) {
    v.process(t.posTimes);
    v.process(t.posValues);
    v.process(t.rotTimes);
    v.process(t.rotValues);
    v.process(t.scaleTimes);
    v.process(t.scaleValues);
}

template <typename V>
void serialize(V& v, AnimationClip& a) {
    v.process(a.name);
    v.process(a.duration);
    v.process(a.ticksPerSecond);
    v.process(a.nodeIndices);
    v.process(a.channels);
}

template <typename V>
void serializeScene(V& v, Scene& s) {
    v.process(s.getMeshes());
    v.process(s.getMaterials());
    v.process(s.getLights());
    v.process(s.getDirectionalLights());
    v.process(s.getAreaLights());
    v.process(s.getNodes());
    v.process(s.getMeshBindings());
    v.process(s.getAnimations());
    v.process(s.getBounds());
    v.process(s.getCamera());
}

// ── Helpers ───────────────────────────────────────────────────

int64_t mtimeOf(const std::filesystem::path& p, std::error_code& ec) {
    auto t = std::filesystem::last_write_time(p, ec);
    if (ec) return 0;
    return (int64_t)t.time_since_epoch().count();
}

bool keysEqual(const SceneCacheKey& a, const SceneCacheKey& b) {
    return a.srcMtime == b.srcMtime
        && a.srcSize  == b.srcSize
        && a.sgMode   == b.sgMode
        && a.emissiveTargetLum == b.emissiveTargetLum;
}

uint64_t fileSize(const std::string& path) {
    std::error_code ec;
    auto sz = std::filesystem::file_size(path, ec);
    return ec ? 0 : (uint64_t)sz;
}

} // namespace

// ── Public API ────────────────────────────────────────────────

std::string SceneCache::cachePathFor(const std::string& srcPath) {
    return srcPath + ".scenecache";
}

bool SceneCache::buildKey(const std::string& srcPath,
                          uint32_t sgMode, float emissiveTargetLum,
                          SceneCacheKey& outKey) {
    std::error_code ec;
    std::filesystem::path p(srcPath);
    auto sz = std::filesystem::file_size(p, ec);
    if (ec) return false;
    int64_t mt = mtimeOf(p, ec);
    if (ec) return false;
    outKey.srcMtime          = mt;
    outKey.srcSize           = (uint64_t)sz;
    outKey.sgMode            = sgMode;
    outKey.emissiveTargetLum = emissiveTargetLum;
    return true;
}

bool SceneCache::tryLoad(const std::string& srcPath, const SceneCacheKey& key, Scene& out) {
    const std::string cachePath = cachePathFor(srcPath);
    std::error_code ec;
    if (!std::filesystem::exists(cachePath, ec) || ec) return false;

    auto t0 = std::chrono::steady_clock::now();

    std::FILE* f = std::fopen(cachePath.c_str(), "rb");
    if (!f) return false;
    Reader r{f};

    uint32_t magic = 0, version = 0;
    uint64_t fingerprint = 0;
    SceneCacheKey storedKey{};
    r.process(magic);
    r.process(version);
    r.process(fingerprint);
    r.process(storedKey);
    if (!r.ok || magic != kCacheMagic || version != kCacheVersion
        || fingerprint != kStructFingerprint || !keysEqual(storedKey, key)) {
        std::fclose(f);
        return false;
    }

    serializeScene(r, out);

    uint32_t magicEnd = 0;
    r.process(magicEnd);
    std::fclose(f);
    if (!r.ok || magicEnd != kCacheMagicEnd) return false;

    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    LOG_INFO("Scene cache hit: %s (%llu bytes, %.1f ms)",
             cachePath.c_str(),
             (unsigned long long)fileSize(cachePath), ms);
    return true;
}

bool SceneCache::save(const std::string& srcPath, const SceneCacheKey& key, const Scene& scene) {
    const std::string cachePath = cachePathFor(srcPath);
    const std::string tmpPath   = cachePath + ".tmp";

    auto t0 = std::chrono::steady_clock::now();

    std::FILE* f = std::fopen(tmpPath.c_str(), "wb");
    if (!f) {
        LOG_WARN("SceneCache: failed to open %s for writing", tmpPath.c_str());
        return false;
    }
    Writer w{f};

    SceneCacheKey keyCopy = key;
    uint32_t magic       = kCacheMagic;
    uint32_t version     = kCacheVersion;
    uint64_t fingerprint = kStructFingerprint;
    w.process(magic);
    w.process(version);
    w.process(fingerprint);
    w.process(keyCopy);

    // serializeScene needs a non-const Scene& for symmetry with Reader; we
    // never mutate here. Cast at the boundary.
    serializeScene(w, const_cast<Scene&>(scene));

    uint32_t magicEnd = kCacheMagicEnd;
    w.process(magicEnd);

    std::fflush(f);
    std::fclose(f);

    if (!w.ok) {
        LOG_WARN("SceneCache: write failed for %s", tmpPath.c_str());
        std::error_code ec;
        std::filesystem::remove(tmpPath, ec);
        return false;
    }

    std::error_code ec;
    std::filesystem::rename(tmpPath, cachePath, ec);
    if (ec) {
        // Windows: rename fails if the target exists. Fall back to overwrite.
        std::filesystem::remove(cachePath, ec);
        std::filesystem::rename(tmpPath, cachePath, ec);
        if (ec) {
            LOG_WARN("SceneCache: rename %s -> %s failed: %s",
                     tmpPath.c_str(), cachePath.c_str(), ec.message().c_str());
            return false;
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    LOG_INFO("Scene cache saved: %s (%llu bytes, %.1f ms)",
             cachePath.c_str(),
             (unsigned long long)fileSize(cachePath), ms);
    return true;
}
