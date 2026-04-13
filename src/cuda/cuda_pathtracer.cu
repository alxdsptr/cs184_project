#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdio>

#include "cuda_types.h"
#include "cuda_math.cuh"
#include "cuda_buffers.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

#define EPS_F 1e-4f
#define INF_F 1e10f

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
        }                                                                  \
    } while (0)

// ============================================================
// Device functions: BBox intersection (slab method)
// ============================================================

__device__ bool bbox_intersect(const CUDARay& ray, float3 bb_min, float3 bb_max) {
    float t0 = ray.min_t;
    float t1 = ray.max_t;

    // X slab
    if (fabsf(ray.d.x) > 1e-12f) {
        float t_near = (bb_min.x - ray.o.x) * ray.inv_d.x;
        float t_far  = (bb_max.x - ray.o.x) * ray.inv_d.x;
        if (t_near > t_far) { float tmp = t_near; t_near = t_far; t_far = tmp; }
        t0 = fmaxf(t0, t_near);
        t1 = fminf(t1, t_far);
        if (t0 > t1) return false;
    } else {
        if (ray.o.x < bb_min.x || ray.o.x > bb_max.x) return false;
    }

    // Y slab
    if (fabsf(ray.d.y) > 1e-12f) {
        float t_near = (bb_min.y - ray.o.y) * ray.inv_d.y;
        float t_far  = (bb_max.y - ray.o.y) * ray.inv_d.y;
        if (t_near > t_far) { float tmp = t_near; t_near = t_far; t_far = tmp; }
        t0 = fmaxf(t0, t_near);
        t1 = fminf(t1, t_far);
        if (t0 > t1) return false;
    } else {
        if (ray.o.y < bb_min.y || ray.o.y > bb_max.y) return false;
    }

    // Z slab
    if (fabsf(ray.d.z) > 1e-12f) {
        float t_near = (bb_min.z - ray.o.z) * ray.inv_d.z;
        float t_far  = (bb_max.z - ray.o.z) * ray.inv_d.z;
        if (t_near > t_far) { float tmp = t_near; t_near = t_far; t_far = tmp; }
        t0 = fmaxf(t0, t_near);
        t1 = fminf(t1, t_far);
        if (t0 > t1) return false;
    } else {
        if (ray.o.z < bb_min.z || ray.o.z > bb_max.z) return false;
    }

    return true;
}

// ============================================================
// Device functions: Triangle intersection (Moller-Trumbore)
// ============================================================

__device__ bool triangle_intersect(CUDARay& ray, const CUDATriangle& tri,
                                    CUDAIntersection& isect) {
    float3 e1 = f3_sub(tri.p2, tri.p1);
    float3 e2 = f3_sub(tri.p3, tri.p1);
    float3 s1 = f3_cross(ray.d, e2);
    float a = f3_dot(e1, s1);

    if (fabsf(a) < 1e-8f) return false;

    float f = 1.0f / a;
    float3 s = f3_sub(ray.o, tri.p1);
    float u = f * f3_dot(s, s1);

    if (u < 0.0f || u > 1.0f) return false;

    float3 s2 = f3_cross(s, e1);
    float v = f * f3_dot(ray.d, s2);

    if (v < 0.0f || u + v > 1.0f) return false;

    float t = f * f3_dot(e2, s2);

    if (t < ray.min_t || t > ray.max_t) return false;

    // Update intersection
    ray.max_t = t;
    isect.t = t;
    isect.hit = 1;
    isect.material_id = tri.material_id;

    // Barycentric interpolation of normal
    float w = 1.0f - u - v;
    isect.n = f3_add(f3_add(f3_mul(tri.n1, w), f3_mul(tri.n2, u)), f3_mul(tri.n3, v));

    return true;
}

// ============================================================
// Device functions: Sphere intersection (quadratic)
// ============================================================

__device__ bool sphere_intersect(CUDARay& ray, const CUDASphere& sph,
                                  CUDAIntersection& isect) {
    float3 oc = f3_sub(ray.o, sph.center);
    float a = f3_dot(ray.d, ray.d);
    float b = 2.0f * f3_dot(oc, ray.d);
    float c = f3_dot(oc, oc) - sph.radius2;
    float disc = b * b - 4.0f * a * c;

    if (disc < 0.0f) return false;

    float sqrt_disc = sqrtf(disc);
    float t1 = (-b - sqrt_disc) / (2.0f * a);
    float t2 = (-b + sqrt_disc) / (2.0f * a);

    float t = t1;
    if (t < ray.min_t || t > ray.max_t) {
        t = t2;
        if (t < ray.min_t || t > ray.max_t) return false;
    }

    ray.max_t = t;
    isect.t = t;
    isect.hit = 1;
    isect.material_id = sph.material_id;

    float3 hit_p = f3_add(ray.o, f3_mul(ray.d, t));
    isect.n = f3_normalize(f3_sub(hit_p, sph.center));

    return true;
}

// ============================================================
// Device functions: BVH traversal (iterative with stack)
// ============================================================

__device__ CUDAIntersection bvh_intersect(CUDARay& ray,
                                           const CUDABVHNode* nodes,
                                           const CUDAPrimitive* prims) {
    CUDAIntersection isect;
    isect.t = ray.max_t;
    isect.hit = 0;

    int stack[64];
    int sp = 0;
    stack[sp++] = 0; // push root

    while (sp > 0) {
        int idx = stack[--sp];
        const CUDABVHNode& node = nodes[idx];

        if (!bbox_intersect(ray, node.bb_min, node.bb_max))
            continue;

        if (node.is_leaf) {
            for (int i = 0; i < node.right_or_count; i++) {
                const CUDAPrimitive& prim = prims[node.left_or_first + i];
                if (prim.type == PRIM_TRIANGLE)
                    triangle_intersect(ray, prim.tri, isect);
                else
                    sphere_intersect(ray, prim.sph, isect);
            }
        } else {
            // Push children - push far child first for better early termination
            stack[sp++] = node.right_or_count;
            stack[sp++] = node.left_or_first;
        }
    }

    return isect;
}

__device__ bool bvh_any_hit(CUDARay& ray,
                             const CUDABVHNode* nodes,
                             const CUDAPrimitive* prims) {
    int stack[64];
    int sp = 0;
    stack[sp++] = 0;

    while (sp > 0) {
        int idx = stack[--sp];
        const CUDABVHNode& node = nodes[idx];

        if (!bbox_intersect(ray, node.bb_min, node.bb_max))
            continue;

        if (node.is_leaf) {
            for (int i = 0; i < node.right_or_count; i++) {
                const CUDAPrimitive& prim = prims[node.left_or_first + i];
                CUDAIntersection dummy;
                dummy.hit = 0;
                bool hit;
                if (prim.type == PRIM_TRIANGLE) {
                    // We make a copy so we don't modify the original ray max_t
                    CUDARay test_ray = ray;
                    hit = triangle_intersect(test_ray, prim.tri, dummy);
                } else {
                    CUDARay test_ray = ray;
                    hit = sphere_intersect(test_ray, prim.sph, dummy);
                }
                if (hit) return true;
            }
        } else {
            stack[sp++] = node.right_or_count;
            stack[sp++] = node.left_or_first;
        }
    }

    return false;
}

// ============================================================
// Device functions: BSDF evaluation
// ============================================================

__device__ float3 bsdf_f(const CUDAMaterial& mat, float3 wo, float3 wi) {
    switch (mat.type) {
        case MAT_DIFFUSE:
            return f3_mul(mat.albedo, 1.0f / M_PI);
        case MAT_EMISSION:
        case MAT_MIRROR:
        case MAT_GLASS:
        case MAT_REFRACTION:
        case MAT_MICROFACET:
        default:
            return f3_zero();
    }
}

__device__ float3 bsdf_sample_f(const CUDAMaterial& mat, float3 wo,
                                 float3* wi, float* pdf, curandState* rng) {
    switch (mat.type) {
        case MAT_DIFFUSE: {
            // Cosine-weighted hemisphere sampling
            float xi1 = curand_uniform(rng);
            float xi2 = curand_uniform(rng);
            float r = sqrtf(xi1);
            float theta = 2.0f * M_PI * xi2;
            float sz = sqrtf(1.0f - xi1);
            *wi = make_float3(r * cosf(theta), r * sinf(theta), sz);
            *pdf = sz / M_PI;
            if (*pdf < 1e-10f) { *pdf = 0.0f; return f3_zero(); }
            return bsdf_f(mat, wo, *wi);
        }
        case MAT_EMISSION: {
            float xi1 = curand_uniform(rng);
            float xi2 = curand_uniform(rng);
            float r = sqrtf(xi1);
            float theta = 2.0f * M_PI * xi2;
            float sz = sqrtf(1.0f - xi1);
            *wi = make_float3(r * cosf(theta), r * sinf(theta), sz);
            *pdf = sz / M_PI;
            return f3_zero(); // emission doesn't scatter
        }
        // Mirror, Glass, Refraction, Microfacet: stubs (matching CPU)
        default:
            *pdf = 0.0f;
            return f3_zero();
    }
}

__device__ float3 bsdf_get_emission(const CUDAMaterial& mat) {
    return mat.emission;
}

// ============================================================
// Device functions: Light sampling
// ============================================================

__device__ float3 light_sample_L(const CUDALight& light, float3 p,
                                  float3* wi, float* dist, float* pdf,
                                  curandState* rng) {
    switch (light.type) {
        case LIGHT_DIRECTIONAL:
            *wi = light.dir_to_light;
            *dist = INF_F;
            *pdf = 1.0f;
            return light.radiance;

        case LIGHT_POINT: {
            float3 d = f3_sub(light.position, p);
            *dist = f3_length(d);
            if (*dist < 1e-8f) { *pdf = 0.0f; return f3_zero(); }
            *wi = f3_mul(d, 1.0f / *dist);
            *pdf = 1.0f;
            return light.radiance;
        }

        case LIGHT_AREA: {
            float sx = curand_uniform(rng) - 0.5f;
            float sy = curand_uniform(rng) - 0.5f;
            float3 light_pos = f3_add(light.position,
                f3_add(f3_mul(light.area_dim_x, sx),
                       f3_mul(light.area_dim_y, sy)));
            float3 d = f3_sub(light_pos, p);
            float sq_dist = f3_dot(d, d);
            *dist = sqrtf(sq_dist);
            if (*dist < 1e-8f) { *pdf = 0.0f; return f3_zero(); }
            *wi = f3_mul(d, 1.0f / *dist);
            float cos_theta = f3_dot(d, light.area_direction);
            float abs_cos = fabsf(cos_theta);
            if (abs_cos < 1e-8f) { *pdf = 0.0f; return f3_zero(); }
            *pdf = sq_dist / (light.area * abs_cos);
            return cos_theta < 0.0f ? light.radiance : f3_zero();
        }

        case LIGHT_SPOT: {
            float3 d = f3_sub(light.position, p);
            *dist = f3_length(d);
            if (*dist < 1e-8f) { *pdf = 0.0f; return f3_zero(); }
            *wi = f3_mul(d, 1.0f / *dist);
            *pdf = 1.0f;
            // Check if point is within spot cone
            float cos_angle = f3_dot(f3_neg(*wi), light.spot_direction);
            if (cos_angle < cosf(light.spot_angle))
                return f3_zero();
            return light.radiance;
        }

        case LIGHT_INF_HEMISPHERE: {
            // Uniform hemisphere sampling
            float xi1 = curand_uniform(rng);
            float xi2 = curand_uniform(rng);
            float theta = acosf(xi1);
            float phi = 2.0f * M_PI * xi2;
            *wi = make_float3(sinf(theta) * cosf(phi),
                              sinf(theta) * sinf(phi),
                              cosf(theta));
            *dist = INF_F;
            *pdf = 1.0f / (2.0f * M_PI);
            return light.radiance;
        }

        default:
            *pdf = 0.0f;
            return f3_zero();
    }
}

// ============================================================
// Device functions: Camera ray generation
// ============================================================

__device__ CUDARay generate_camera_ray(const CUDACameraParams& cam,
                                        float px, float py) {
    // px, py are normalized [0,1] sensor coordinates
    float camera_x = (px - 0.5f) * cam.hFov_half_tan * 2.0f;
    float camera_y = (py - 0.5f) * cam.vFov_half_tan * 2.0f;
    float3 dir_camera = make_float3(camera_x, camera_y, -1.0f);

    // Transform to world space using c2w matrix
    CUDAMatrix3x3 c2w;
    c2w.col0 = cam.c2w_col0;
    c2w.col1 = cam.c2w_col1;
    c2w.col2 = cam.c2w_col2;
    float3 dir_world = f3_normalize(mat3_mul_vec(c2w, dir_camera));

    CUDARay ray;
    ray.o = cam.pos;
    ray.d = dir_world;
    ray.inv_d = make_float3(1.0f / dir_world.x, 1.0f / dir_world.y, 1.0f / dir_world.z);
    ray.min_t = cam.nClip;
    ray.max_t = cam.fClip;
    ray.depth = 0;
    return ray;
}

// ============================================================
// Kernel: Initialize per-pixel RNG states
// ============================================================

__global__ void init_rng_kernel(curandState* states, int w, int h, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= w * h) return;
    curand_init(seed, idx, 0, &states[idx]);
}

// ============================================================
// Kernel: Main path tracing kernel
// ============================================================

__global__ void pathtrace_kernel(
    float*               d_hdr_buffer,
    int*                 d_sample_count,
    curandState*         d_rng_states,
    const CUDABVHNode*   d_bvh,
    const CUDAPrimitive* d_prims,
    const CUDAMaterial*  d_materials,
    const CUDALight*     d_lights,
    CUDACameraParams     cam,
    CUDARenderParams     params)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= cam.screenW || py >= cam.screenH) return;

    int pixel_idx = px + py * cam.screenW;
    curandState rng = d_rng_states[pixel_idx];

    float3 radiance_sum = f3_zero();
    float s1 = 0.0f, s2 = 0.0f;
    int actual_samples = 0;

    int ns_aa = params.ns_aa;
    int max_depth = params.max_ray_depth;
    int ns_area_light = params.ns_area_light;
    int is_accum = params.is_accum_bounces;
    int use_hemisphere = params.direct_hemisphere_sample;
    int batch_size = params.samples_per_batch > 0 ? params.samples_per_batch : 1;
    float tolerance = params.max_tolerance;
    int num_lights = params.num_lights;

    for (int s = 0; s < ns_aa; s++) {
        // Jittered sub-pixel sample
        float sx = (float)px + curand_uniform(&rng);
        float sy = (float)py + curand_uniform(&rng);

        CUDARay ray = generate_camera_ray(cam,
            sx / (float)cam.screenW, sy / (float)cam.screenH);

        // ---- Iterative path tracing ----
        float3 path_radiance = f3_zero();
        float3 throughput = make_float3(1.0f, 1.0f, 1.0f);

        for (int bounce = 0; bounce <= max_depth; bounce++) {
            CUDARay trace_ray = ray;
            CUDAIntersection isect = bvh_intersect(trace_ray, d_bvh, d_prims);
            ray.max_t = trace_ray.max_t; // propagate updated max_t

            if (!isect.hit) {
                // No intersection - could add environment light here
                break;
            }

            const CUDAMaterial& mat = d_materials[isect.material_id];

            // Zero bounce: add emission
            if (bounce == 0) {
                path_radiance = f3_add(path_radiance,
                    f3_mul_comp(throughput, bsdf_get_emission(mat)));
            }

            if (bounce >= max_depth) break;

            // Compute hit point and local coordinate system
            float3 hit_p = f3_add(ray.o, f3_mul(ray.d, isect.t));
            CUDAMatrix3x3 o2w;
            make_coord_space_device(o2w, isect.n);
            CUDAMatrix3x3 w2o = mat3_transpose(o2w);
            float3 w_out = mat3_mul_vec(w2o, f3_neg(ray.d));

            // ---- Direct lighting ----
            float3 direct = f3_zero();

            if (!use_hemisphere) {
                // Importance sampling over lights
                for (int li = 0; li < num_lights; li++) {
                    int n_samples = d_lights[li].is_delta ? 1 : ns_area_light;
                    float3 light_contrib = f3_zero();

                    for (int ls = 0; ls < n_samples; ls++) {
                        float3 wi_world;
                        float dist_to_light, pdf;
                        float3 L_i = light_sample_L(d_lights[li], hit_p,
                            &wi_world, &dist_to_light, &pdf, &rng);
                        if (pdf <= 0.0f) continue;

                        float3 wi = mat3_mul_vec(w2o, wi_world);
                        if (wi.z <= 0.0f) continue;

                        // Shadow ray
                        CUDARay shadow_ray;
                        shadow_ray.o = hit_p;
                        shadow_ray.d = wi_world;
                        shadow_ray.inv_d = make_float3(
                            1.0f / wi_world.x, 1.0f / wi_world.y, 1.0f / wi_world.z);
                        shadow_ray.min_t = EPS_F;
                        shadow_ray.max_t = dist_to_light - EPS_F;

                        if (!bvh_any_hit(shadow_ray, d_bvh, d_prims)) {
                            float3 bsdf_val = bsdf_f(mat, w_out, wi);
                            float cos_t = fabsf(wi.z);
                            light_contrib = f3_add(light_contrib,
                                f3_mul(f3_mul_comp(bsdf_val, L_i), cos_t / pdf));
                        }
                    }
                    direct = f3_add(direct, f3_mul(light_contrib, 1.0f / (float)n_samples));
                }
            } else {
                // Hemisphere sampling
                int num_samples = num_lights * ns_area_light;
                if (num_samples > 0) {
                    float hemisphere_pdf = 1.0f / (2.0f * M_PI);
                    for (int hs = 0; hs < num_samples; hs++) {
                        // Uniform hemisphere sample
                        float xi1 = curand_uniform(&rng);
                        float xi2 = curand_uniform(&rng);
                        float theta = acosf(xi1);
                        float phi = 2.0f * M_PI * xi2;
                        float3 wi = make_float3(
                            sinf(theta) * cosf(phi),
                            sinf(theta) * sinf(phi),
                            cosf(theta));

                        float3 wi_world = mat3_mul_vec(o2w, wi);

                        CUDARay shadow_ray;
                        shadow_ray.o = hit_p;
                        shadow_ray.d = wi_world;
                        shadow_ray.inv_d = make_float3(
                            1.0f / wi_world.x, 1.0f / wi_world.y, 1.0f / wi_world.z);
                        shadow_ray.min_t = EPS_F;
                        shadow_ray.max_t = INF_F;

                        CUDAIntersection light_isect = bvh_intersect(shadow_ray, d_bvh, d_prims);
                        if (light_isect.hit) {
                            float3 L_i = bsdf_get_emission(d_materials[light_isect.material_id]);
                            float3 bsdf_val = bsdf_f(mat, w_out, wi);
                            float cos_t = fabsf(wi.z);
                            direct = f3_add(direct,
                                f3_mul(f3_mul_comp(bsdf_val, L_i), cos_t / hemisphere_pdf));
                        }
                    }
                    direct = f3_mul(direct, 1.0f / (float)num_samples);
                }
            }

            // Accumulate direct lighting
            if (is_accum || bounce == 0) {
                path_radiance = f3_add(path_radiance,
                    f3_mul_comp(throughput, direct));
            }

            // Russian roulette (skip for first indirect bounce)
            float continue_prob = 0.65f;
            bool force_first = (bounce == 0);
            if (!force_first) {
                if (curand_uniform(&rng) >= continue_prob)
                    break;
                throughput = f3_mul(throughput, 1.0f / continue_prob);
            }

            // Sample next direction from BSDF
            float3 wi_local;
            float pdf;
            float3 f = bsdf_sample_f(mat, w_out, &wi_local, &pdf, &rng);
            if (pdf <= 0.0f || wi_local.z <= 0.0f) break;

            // Update throughput
            float cos_t = fabsf(wi_local.z);
            throughput = f3_mul_comp(throughput, f3_mul(f, cos_t / pdf));

            // Spawn next ray
            float3 wi_world = mat3_mul_vec(o2w, wi_local);
            ray.o = hit_p;
            ray.d = wi_world;
            ray.inv_d = make_float3(1.0f / wi_world.x, 1.0f / wi_world.y, 1.0f / wi_world.z);
            ray.min_t = EPS_F;
            ray.max_t = INF_F;
        }
        // ---- End iterative path tracing ----

        radiance_sum = f3_add(radiance_sum, path_radiance);

        float illum = 0.2126f * path_radiance.x
                    + 0.7152f * path_radiance.y
                    + 0.0722f * path_radiance.z;
        s1 += illum;
        s2 += illum * illum;
        actual_samples = s + 1;

        // Adaptive sampling check
        if (actual_samples % batch_size == 0 && actual_samples > 1) {
            float n = (float)actual_samples;
            float mean = s1 / n;
            float variance = (s2 - (s1 * s1) / n) / (n - 1.0f);
            if (variance < 0.0f) variance = 0.0f;
            float I = 1.96f * sqrtf(variance / n);
            if (I <= tolerance * mean) break;
        }
    }

    // Write results
    float inv_n = (actual_samples > 0) ? (1.0f / (float)actual_samples) : 0.0f;
    d_hdr_buffer[pixel_idx * 3 + 0] = radiance_sum.x * inv_n;
    d_hdr_buffer[pixel_idx * 3 + 1] = radiance_sum.y * inv_n;
    d_hdr_buffer[pixel_idx * 3 + 2] = radiance_sum.z * inv_n;
    d_sample_count[pixel_idx] = actual_samples;

    // Save RNG state back
    d_rng_states[pixel_idx] = rng;
}

// ============================================================
// Host functions: GPU memory management
// ============================================================

void cuda_upload_scene(CUDASceneBuffers& buf,
                       const CUDABVHNode* bvh_nodes, int num_nodes,
                       const CUDAPrimitive* primitives, int num_prims,
                       const CUDAMaterial* materials, int num_mats,
                       const CUDALight* lights, int num_lights)
{
    // Free previous scene data if any
    if (buf.d_bvh_nodes) cudaFree(buf.d_bvh_nodes);
    if (buf.d_primitives) cudaFree(buf.d_primitives);
    if (buf.d_materials) cudaFree(buf.d_materials);
    if (buf.d_lights) cudaFree(buf.d_lights);

    buf.num_bvh_nodes = num_nodes;
    buf.num_primitives = num_prims;
    buf.num_materials = num_mats;
    buf.num_lights = num_lights;

    CUDA_CHECK(cudaMalloc(&buf.d_bvh_nodes, num_nodes * sizeof(CUDABVHNode)));
    CUDA_CHECK(cudaMemcpy(buf.d_bvh_nodes, bvh_nodes,
        num_nodes * sizeof(CUDABVHNode), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&buf.d_primitives, num_prims * sizeof(CUDAPrimitive)));
    CUDA_CHECK(cudaMemcpy(buf.d_primitives, primitives,
        num_prims * sizeof(CUDAPrimitive), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&buf.d_materials, num_mats * sizeof(CUDAMaterial)));
    CUDA_CHECK(cudaMemcpy(buf.d_materials, materials,
        num_mats * sizeof(CUDAMaterial), cudaMemcpyHostToDevice));

    if (num_lights > 0) {
        CUDA_CHECK(cudaMalloc(&buf.d_lights, num_lights * sizeof(CUDALight)));
        CUDA_CHECK(cudaMemcpy(buf.d_lights, lights,
            num_lights * sizeof(CUDALight), cudaMemcpyHostToDevice));
    }

    fprintf(stdout, "[CUDA] Uploaded scene: %d BVH nodes, %d primitives, %d materials, %d lights\n",
            num_nodes, num_prims, num_mats, num_lights);
}

void cuda_alloc_output(CUDASceneBuffers& buf, int w, int h) {
    // Free previous output buffers
    if (buf.d_hdr_buffer) cudaFree(buf.d_hdr_buffer);
    if (buf.d_sample_count) cudaFree(buf.d_sample_count);
    if (buf.d_rng_states) cudaFree(buf.d_rng_states);

    buf.output_w = w;
    buf.output_h = h;

    CUDA_CHECK(cudaMalloc(&buf.d_hdr_buffer, w * h * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buf.d_sample_count, w * h * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&buf.d_rng_states, w * h * sizeof(curandState)));

    // Initialize RNG states
    int total_pixels = w * h;
    int block = 256;
    int grid = (total_pixels + block - 1) / block;
    init_rng_kernel<<<grid, block>>>(buf.d_rng_states, w, h, 42ULL);
    CUDA_CHECK(cudaDeviceSynchronize());

    fprintf(stdout, "[CUDA] Allocated output buffers: %dx%d\n", w, h);
}

void cuda_free_all(CUDASceneBuffers& buf) {
    if (buf.d_bvh_nodes)    { cudaFree(buf.d_bvh_nodes);    buf.d_bvh_nodes = nullptr; }
    if (buf.d_primitives)   { cudaFree(buf.d_primitives);   buf.d_primitives = nullptr; }
    if (buf.d_materials)    { cudaFree(buf.d_materials);    buf.d_materials = nullptr; }
    if (buf.d_lights)       { cudaFree(buf.d_lights);       buf.d_lights = nullptr; }
    if (buf.d_hdr_buffer)   { cudaFree(buf.d_hdr_buffer);   buf.d_hdr_buffer = nullptr; }
    if (buf.d_sample_count) { cudaFree(buf.d_sample_count); buf.d_sample_count = nullptr; }
    if (buf.d_rng_states)   { cudaFree(buf.d_rng_states);   buf.d_rng_states = nullptr; }
}

void cuda_render(CUDASceneBuffers& buf,
                 const CUDACameraParams& cam,
                 const CUDARenderParams& params)
{
    int w = cam.screenW;
    int h = cam.screenH;

    // Clear output
    CUDA_CHECK(cudaMemset(buf.d_hdr_buffer, 0, w * h * 3 * sizeof(float)));
    CUDA_CHECK(cudaMemset(buf.d_sample_count, 0, w * h * sizeof(int)));

    // Launch kernel
    dim3 block(16, 16);
    dim3 grid((w + 15) / 16, (h + 15) / 16);

    pathtrace_kernel<<<grid, block>>>(
        buf.d_hdr_buffer,
        buf.d_sample_count,
        buf.d_rng_states,
        buf.d_bvh_nodes,
        buf.d_primitives,
        buf.d_materials,
        buf.d_lights,
        cam,
        params
    );

    CUDA_CHECK(cudaDeviceSynchronize());
}

void cuda_download_results(const CUDASceneBuffers& buf,
                           float* host_hdr, int* host_sample_count,
                           int w, int h)
{
    CUDA_CHECK(cudaMemcpy(host_hdr, buf.d_hdr_buffer,
        w * h * 3 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_sample_count, buf.d_sample_count,
        w * h * sizeof(int), cudaMemcpyDeviceToHost));
}
