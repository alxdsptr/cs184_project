#include "accel/SAH_BVH.h"
#include "util/Log.h"
#include <algorithm>
#include <numeric>
#include <cstring>

static constexpr int NUM_BINS = 12;
static constexpr int MAX_LEAF_PRIMS = 4;

struct PrimInfo {
    AABB  bounds;
    float3 centroid;
    uint32_t origIndex;
};

struct Bucket {
    AABB     bounds;
    uint32_t count = 0;
};

// Recursive SAH build, returns index of root node in `nodes`
static uint32_t buildRecursive(
    std::vector<BVHNode>& nodes,
    std::vector<PrimInfo>& prims,
    std::vector<uint32_t>& orderedPrims,
    uint32_t start, uint32_t end)
{
    uint32_t nPrims = end - start;

    // Compute bounds of all primitives and centroids
    AABB totalBounds, centroidBounds;
    for (uint32_t i = start; i < end; i++) {
        totalBounds.expand(prims[i].bounds);
        centroidBounds.expand(prims[i].centroid);
    }

    // Create leaf if few enough primitives
    if (nPrims <= MAX_LEAF_PRIMS) {
        BVHNode leaf;
        leaf.bounds = totalBounds;
        leaf.primOffset = (uint32_t)orderedPrims.size();
        leaf.primCount  = nPrims;
        for (uint32_t i = start; i < end; i++)
            orderedPrims.push_back(prims[i].origIndex);
        uint32_t idx = (uint32_t)nodes.size();
        nodes.push_back(leaf);
        return idx;
    }

    // Find best split axis and position via SAH with binning
    float3 extent = centroidBounds.bmax - centroidBounds.bmin;
    int bestAxis = 0;
    if (extent.y > extent.x && extent.y > extent.z) bestAxis = 1;
    else if (extent.z > extent.x && extent.z > extent.y) bestAxis = 2;

    float axisExtent = (bestAxis == 0) ? extent.x : (bestAxis == 1) ? extent.y : extent.z;

    // Degenerate case: all centroids at same point
    if (axisExtent < 1e-7f) {
        BVHNode leaf;
        leaf.bounds = totalBounds;
        leaf.primOffset = (uint32_t)orderedPrims.size();
        leaf.primCount  = nPrims;
        for (uint32_t i = start; i < end; i++)
            orderedPrims.push_back(prims[i].origIndex);
        uint32_t idx = (uint32_t)nodes.size();
        nodes.push_back(leaf);
        return idx;
    }

    float cmin = (bestAxis == 0) ? centroidBounds.bmin.x
               : (bestAxis == 1) ? centroidBounds.bmin.y
               : centroidBounds.bmin.z;

    // Initialize bins
    Bucket bins[NUM_BINS];
    for (uint32_t i = start; i < end; i++) {
        float c = (bestAxis == 0) ? prims[i].centroid.x
                : (bestAxis == 1) ? prims[i].centroid.y
                : prims[i].centroid.z;
        int b = (int)(NUM_BINS * ((c - cmin) / axisExtent));
        if (b >= NUM_BINS) b = NUM_BINS - 1;
        bins[b].count++;
        bins[b].bounds.expand(prims[i].bounds);
    }

    // Evaluate SAH cost for each split
    float bestCost = 1e30f;
    int bestSplit = -1;

    // Precompute prefix/suffix bounds and counts while skipping empty bins.
    // This avoids expanding with default-initialized invalid AABBs from empty bins.
    AABB prefixBounds[NUM_BINS];
    AABB suffixBounds[NUM_BINS];
    uint32_t prefixCount[NUM_BINS] = {};
    uint32_t suffixCount[NUM_BINS] = {};

    AABB runningLeft;
    uint32_t runningLeftCount = 0;
    for (int i = 0; i < NUM_BINS; i++) {
        if (bins[i].count > 0) {
            runningLeft.expand(bins[i].bounds);
            runningLeftCount += bins[i].count;
        }
        prefixBounds[i] = runningLeft;
        prefixCount[i] = runningLeftCount;
    }

    AABB runningRight;
    uint32_t runningRightCount = 0;
    for (int i = NUM_BINS - 1; i >= 0; i--) {
        if (bins[i].count > 0) {
            runningRight.expand(bins[i].bounds);
            runningRightCount += bins[i].count;
        }
        suffixBounds[i] = runningRight;
        suffixCount[i] = runningRightCount;
    }

    for (int s = 0; s < NUM_BINS - 1; s++) {
        const AABB& bLeft = prefixBounds[s];
        const AABB& bRight = suffixBounds[s + 1];
        uint32_t cntLeft = prefixCount[s];
        uint32_t cntRight = suffixCount[s + 1];
        if (cntLeft == 0 || cntRight == 0) continue;
        float totalSA = totalBounds.surfaceArea();
        if (totalSA < 1e-20f) {
            continue;
        }
        float cost = 0.125f + (cntLeft * bLeft.surfaceArea() + cntRight * bRight.surfaceArea())
                     / totalSA;
        if (cost < bestCost) {
            bestCost  = cost;
            bestSplit = s;
        }
    }

    // If SAH split is worse than making a leaf, make a leaf
    float leafCost = (float)nPrims;
    if (bestSplit == -1 || (nPrims <= 2 * MAX_LEAF_PRIMS && bestCost > leafCost)) {
        BVHNode leaf;
        leaf.bounds = totalBounds;
        leaf.primOffset = (uint32_t)orderedPrims.size();
        leaf.primCount  = nPrims;
        for (uint32_t i = start; i < end; i++)
            orderedPrims.push_back(prims[i].origIndex);
        uint32_t idx = (uint32_t)nodes.size();
        nodes.push_back(leaf);
        return idx;
    }

    // Partition primitives (manual partition to avoid MSVC ICE with std::partition lambda)
    uint32_t mid = start;
    for (uint32_t i = start; i < end; i++) {
        float c = (bestAxis == 0) ? prims[i].centroid.x
                : (bestAxis == 1) ? prims[i].centroid.y
                : prims[i].centroid.z;
        int b = (int)(NUM_BINS * ((c - cmin) / axisExtent));
        if (b >= NUM_BINS) b = NUM_BINS - 1;
        if (b <= bestSplit) {
            std::swap(prims[i], prims[mid]);
            mid++;
        }
    }

    // Safety: ensure partition actually split
    if (mid == start || mid == end) {
        mid = (start + end) / 2;
    }

    // Reserve node slot (interior)
    uint32_t nodeIdx = (uint32_t)nodes.size();
    BVHNode interiorNode;
    memset(&interiorNode, 0, sizeof(BVHNode));
    nodes.push_back(interiorNode);

    uint32_t leftIdx  = buildRecursive(nodes, prims, orderedPrims, start, mid);
    uint32_t rightIdx = buildRecursive(nodes, prims, orderedPrims, mid, end);

    nodes[nodeIdx].bounds     = totalBounds;
    nodes[nodeIdx].leftChild  = leftIdx;
    nodes[nodeIdx].rightChild = rightIdx;
    nodes[nodeIdx].primCount  = 0; // mark as interior

    return nodeIdx;
}

BVHData SAH_BVH::build(const float3* positions, const uint32_t* indices, uint32_t triCount) {
    BVHData result;

    if (triCount == 0) {
        LOG_WARN("SAH_BVH: 0 triangles");
        return result;
    }

    // Build per-triangle info
    std::vector<PrimInfo> prims(triCount);
    for (uint32_t i = 0; i < triCount; i++) {
        float3 v0 = positions[indices[i*3+0]];
        float3 v1 = positions[indices[i*3+1]];
        float3 v2 = positions[indices[i*3+2]];
        prims[i].bounds = AABB();
        prims[i].bounds.expand(v0);
        prims[i].bounds.expand(v1);
        prims[i].bounds.expand(v2);
        prims[i].centroid = prims[i].bounds.center();
        prims[i].origIndex = i;
    }

    result.rootIndex = buildRecursive(result.nodes, prims, result.orderedPrimIndices, 0, triCount);

    LOG_INFO("SAH BVH: %u nodes, %u triangles, root=%u",
             (uint32_t)result.nodes.size(), triCount, result.rootIndex);

    return result;
}
