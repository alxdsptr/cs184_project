#ifndef CGL_CUDA_TYPES_H
#define CGL_CUDA_TYPES_H

// GPU-side data structures for CUDA path tracer.
// This header is included by both .cpp and .cu files.
// Uses float (not double) for GPU performance.

#ifndef __CUDACC__
// When compiled by regular C++ compiler, define float3/int2 equivalents
struct float3 { float x, y, z; };
inline float3 make_float3(float x, float y, float z) { return {x, y, z}; }
#endif

// ============================================================
// Ray
// ============================================================
struct CUDARay {
    float3 o;       // origin
    float3 d;       // direction
    float3 inv_d;   // 1/d precomputed
    float  min_t;
    float  max_t;
    int    depth;
};

// ============================================================
// Intersection result
// ============================================================
struct CUDAIntersection {
    float  t;
    float3 n;           // interpolated normal
    int    material_id;  // index into material array
    int    hit;          // 0 or 1
};

// ============================================================
// Linearized BVH node
// ============================================================
struct CUDABVHNode {
    float3 bb_min;
    float3 bb_max;
    int    left_or_first;   // inner: left child index; leaf: first primitive index
    int    right_or_count;  // inner: right child index; leaf: primitive count
    int    is_leaf;
    int    pad;             // padding for alignment
};

// ============================================================
// Primitives (tagged union)
// ============================================================
enum CUDAPrimType { PRIM_TRIANGLE = 0, PRIM_SPHERE = 1 };

struct CUDATriangle {
    float3 p1, p2, p3;     // vertices
    float3 n1, n2, n3;     // vertex normals
    int    material_id;
    int    pad;
};

struct CUDASphere {
    float3 center;
    float  radius;
    float  radius2;         // radius squared
    int    material_id;
    int    pad;
};

struct CUDAPrimitive {
    int type;  // CUDAPrimType
    int pad;
    union {
        CUDATriangle tri;
        CUDASphere   sph;
    };
};

// ============================================================
// Material (tagged union replacing BSDF hierarchy)
// ============================================================
enum CUDAMaterialType {
    MAT_DIFFUSE = 0,
    MAT_EMISSION = 1,
    MAT_MIRROR = 2,
    MAT_GLASS = 3,
    MAT_REFRACTION = 4,
    MAT_MICROFACET = 5
};

struct CUDAMaterial {
    int    type;        // CUDAMaterialType
    int    is_delta;

    float3 albedo;      // reflectance (diffuse, mirror)
    float3 emission;    // emitted radiance

    // Glass / Refraction
    float  ior;
    float3 transmittance;

    // Microfacet
    float3 eta;         // complex IOR real part
    float3 k;           // complex IOR imaginary part
    float  alpha;       // roughness

    float  pad;
};

// ============================================================
// Light (tagged union replacing SceneLight hierarchy)
// ============================================================
enum CUDALightType {
    LIGHT_DIRECTIONAL = 0,
    LIGHT_POINT = 1,
    LIGHT_AREA = 2,
    LIGHT_SPOT = 3,
    LIGHT_INF_HEMISPHERE = 4
};

struct CUDALight {
    int    type;        // CUDALightType
    int    is_delta;

    float3 radiance;

    // Directional
    float3 dir_to_light;

    // Point / Spot
    float3 position;

    // Spot
    float3 spot_direction;
    float  spot_angle;

    // Area
    float3 area_direction;  // surface normal
    float3 area_dim_x;
    float3 area_dim_y;
    float  area;

    float  pad;
};

// ============================================================
// Camera parameters (flat struct)
// ============================================================
struct CUDACameraParams {
    float3 pos;
    float3 c2w_col0;    // camera-to-world matrix column 0
    float3 c2w_col1;    // column 1
    float3 c2w_col2;    // column 2
    float  hFov_half_tan;
    float  vFov_half_tan;
    float  nClip;
    float  fClip;
    float  lensRadius;
    float  focalDistance;
    int    screenW;
    int    screenH;
};

// ============================================================
// Render parameters
// ============================================================
struct CUDARenderParams {
    int  ns_aa;
    int  max_ray_depth;
    int  ns_area_light;
    int  is_accum_bounces;
    int  direct_hemisphere_sample;
    int  samples_per_batch;
    float max_tolerance;
    int  num_lights;
};

#endif // CGL_CUDA_TYPES_H
