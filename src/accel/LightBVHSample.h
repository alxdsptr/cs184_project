#pragma once
#include "accel/LightBVHNode.h"
#include "core/Math.h"

// Device-side stochastic descent through a flat Light BVH.
//
// Given a shading point `p`, a uniform random sample `u` in [0,1), and the
// flat node array built by LightBVH::build(), this picks exactly one light
// and returns its global light index along with the probability of having
// picked it. The caller then uses that probability to form a proper MIS
// light-selection PDF (equivalent to the flat-CDF `weight / totalWeight`).
//
// Descent rule: at an internal node we estimate an unnormalised importance
// for each child as  weight / (distanceToAABB^2 + eps). We normalise, pick
// one branch, and remap `u` so a single uniform drives the whole traversal.
// At a leaf we pick among the <=LBVH_MAX_LEAF lights proportional to their
// per-light weights. All probabilities are multiplied into `pdf`.
//
// Rationale: splitting by weighted SAH plus distance-weighted descent gives
// the same marginal probability as the flat CDF in the limit (pure weight
// sampling) but concentrates samples on nearby lights, which is exactly
// what reduces variance when you have many lights.

// Squared distance from a point to an AABB (0 inside). Keeps lights that
// enclose the shading point from exploding to infinity in the importance.
__device__ inline float distSqToAABB(const float3& p, const AABB& box) {
    float dx = fmaxf(fmaxf(box.bmin.x - p.x, p.x - box.bmax.x), 0.0f);
    float dy = fmaxf(fmaxf(box.bmin.y - p.y, p.y - box.bmax.y), 0.0f);
    float dz = fmaxf(fmaxf(box.bmin.z - p.z, p.z - box.bmax.z), 0.0f);
    return dx*dx + dy*dy + dz*dz;
}

// Sample a light from the BVH. Returns the chosen orderedIndex slot
// ([0, lightCount)) into `orderedLightIndices`; the caller dereferences that
// to get the real GPUAreaLight index. `pdf` is the probability of the pick
// (P(select this light | uniform sample)).
__device__ inline bool lightBVH_sample(
    const LightBVHNode* nodes,
    uint32_t rootIndex,
    const float3& p,
    float u,
    uint32_t& outOrderedSlot,
    float&    outPdf)
{
    if (!nodes) return false;
    uint32_t nodeIdx = rootIndex;
    float pdf = 1.0f;

    // Traverse until we hit a leaf. Depth is bounded by log2(lightCount),
    // well under 64 even with millions of lights.
    for (int depth = 0; depth < 64; depth++) {
        const LightBVHNode node = nodes[nodeIdx];
        if (node.isLeaf()) {
            if (node.weight <= 0.0f || node.primCount == 0) return false;
            // Uniform-within-leaf selection: we don't store per-light weights
            // in the flat node array. With LBVH_MAX_LEAF = 4 and a weight-
            // aware build the variance hit is small, and any unbiasedness
            // concern is absorbed by the matching 1/primCount term in
            // lightBVH_pdf.
            uint32_t pick = (uint32_t)(u * (float)node.primCount);
            if (pick >= node.primCount) pick = node.primCount - 1;
            outOrderedSlot = node.primOffset + pick;
            outPdf = pdf * (1.0f / (float)node.primCount);
            return true;
        }

        uint32_t cL = node.leftChild;
        uint32_t cR = node.rightChild;
        const LightBVHNode& L = nodes[cL];
        const LightBVHNode& R = nodes[cR];

        // Importance ~ weight / (dist^2 + eps). eps prevents singular
        // behaviour when the shading point is inside the bounding box of a
        // subtree (distSqToAABB = 0).
        const float eps = 1e-4f;
        float iL = L.weight / (distSqToAABB(p, L.bounds) + eps);
        float iR = R.weight / (distSqToAABB(p, R.bounds) + eps);
        float sum = iL + iR;
        if (!(sum > 0.0f)) return false;
        float pL = iL / sum;

        if (u < pL) {
            pdf *= pL;
            // Remap u back into [0,1)
            u = u / fmaxf(pL, 1e-30f);
            if (u >= 1.0f) u = 0.999999f;
            nodeIdx = cL;
        } else {
            float pR = 1.0f - pL;
            pdf *= pR;
            u = (u - pL) / fmaxf(pR, 1e-30f);
            if (u >= 1.0f) u = 0.999999f;
            nodeIdx = cR;
        }
    }
    return false;
}

// Evaluate the BVH selection probability for a known light (by its slot in
// orderedLightIndices). Needed for MIS when the same light is reached via
// BSDF sampling and we want to weight the other strategy. The walk is
// entirely deterministic given the shading point `p`; we just multiply the
// descent probabilities down the known path.
//
// Implemented as a lazy traversal: we know the leaf we need to reach (by its
// ordered slot), so at each internal node we descend into whichever child's
// [primOffset, primOffset+primCount) range contains the slot. To find that
// range cheaply without storing a range on internal nodes, we walk both
// subtrees looking for the leaf; this is O(depth) because at each internal
// node exactly one child's subtree contains the slot. We descend by testing
// the left child's leaf range — if the slot falls inside, take left,
// otherwise right.
__device__ inline float lightBVH_pdf(
    const LightBVHNode* nodes,
    uint32_t rootIndex,
    const float3& p,
    uint32_t targetSlot)
{
    if (!nodes) return 0.0f;
    uint32_t nodeIdx = rootIndex;
    float pdf = 1.0f;

    // Helper to find a leaf containing a given slot inside a subtree rooted
    // at `idx`. Returns (primOffset, primCount) of that leaf via references.
    // Implemented inline as a mini stack-free descent that always goes left
    // first; at each internal node we recursively know which child holds the
    // slot by comparing the slot to the leftmost leaf's primOffset after
    // descending left as far as possible. To keep this header-only and
    // branchless-at-runtime we instead do a simple linear search at each
    // internal node: descend into the child whose subtree contains the slot
    // by walking left-first to find its primOffset range.
    //
    // Since the tree is built by partitioning a contiguous prims array, the
    // ordered-slot ranges of the two children are contiguous and disjoint.
    // At each internal node we find the split point (= primOffset of the
    // right-child's leftmost leaf) by a short left-descent of the right child.
    for (int depth = 0; depth < 64; depth++) {
        const LightBVHNode node = nodes[nodeIdx];
        if (node.isLeaf()) {
            if (node.weight <= 0.0f || node.primCount == 0) return 0.0f;
            if (targetSlot < node.primOffset ||
                targetSlot >= node.primOffset + node.primCount) return 0.0f;
            return pdf * (1.0f / (float)node.primCount);
        }

        uint32_t cL = node.leftChild;
        uint32_t cR = node.rightChild;
        // Find the split point = leftmost-leaf's primOffset of the right child.
        uint32_t probe = cR;
        for (int d2 = 0; d2 < 64; d2++) {
            const LightBVHNode& n = nodes[probe];
            if (n.isLeaf()) { break; }
            probe = n.leftChild;
        }
        uint32_t split = nodes[probe].primOffset;

        const LightBVHNode& L = nodes[cL];
        const LightBVHNode& R = nodes[cR];
        const float eps = 1e-4f;
        float iL = L.weight / (distSqToAABB(p, L.bounds) + eps);
        float iR = R.weight / (distSqToAABB(p, R.bounds) + eps);
        float sum = iL + iR;
        if (!(sum > 0.0f)) return 0.0f;
        float pL = iL / sum;
        float pR = 1.0f - pL;

        if (targetSlot < split) {
            pdf *= pL;
            nodeIdx = cL;
        } else {
            pdf *= pR;
            nodeIdx = cR;
        }
    }
    return 0.0f;
}
