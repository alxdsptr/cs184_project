#include "accel/LightBVH.h"
#include "util/Log.h"
#include <algorithm>
#include <cstring>

static constexpr int   LBVH_NUM_BINS     = 12;
static constexpr int   LBVH_MAX_LEAF     = 4;

// Per-light record used during construction.
struct LBPrim {
    AABB     bounds;
    float3   centroid;
    float    weight;
    uint32_t origIndex;
};

struct LBBin {
    AABB     bounds;
    float    weight = 0.0f;
    uint32_t count  = 0;
};

static uint32_t buildRecursive(
    std::vector<LightBVHNode>& nodes,
    std::vector<LBPrim>&       prims,
    std::vector<uint32_t>&     orderedPrims,
    uint32_t start, uint32_t end)
{
    uint32_t nPrims = end - start;

    AABB  totalBounds, centroidBounds;
    float totalWeight = 0.0f;
    for (uint32_t i = start; i < end; i++) {
        totalBounds.expand(prims[i].bounds);
        centroidBounds.expand(prims[i].centroid);
        totalWeight += prims[i].weight;
    }

    auto makeLeaf = [&]() {
        LightBVHNode leaf;
        leaf.bounds     = totalBounds;
        leaf.weight     = totalWeight;
        leaf.primOffset = (uint32_t)orderedPrims.size();
        leaf.primCount  = nPrims;
        for (uint32_t i = start; i < end; i++)
            orderedPrims.push_back(prims[i].origIndex);
        uint32_t idx = (uint32_t)nodes.size();
        nodes.push_back(leaf);
        return idx;
    };

    if (nPrims <= LBVH_MAX_LEAF) return makeLeaf();

    float3 extent = centroidBounds.bmax - centroidBounds.bmin;
    int bestAxis = 0;
    if (extent.y > extent.x && extent.y > extent.z) bestAxis = 1;
    else if (extent.z > extent.x && extent.z > extent.y) bestAxis = 2;
    float axisExtent = (bestAxis == 0) ? extent.x
                     : (bestAxis == 1) ? extent.y
                                       : extent.z;
    if (axisExtent < 1e-7f) return makeLeaf();

    float cmin = (bestAxis == 0) ? centroidBounds.bmin.x
               : (bestAxis == 1) ? centroidBounds.bmin.y
                                 : centroidBounds.bmin.z;

    LBBin bins[LBVH_NUM_BINS];
    for (uint32_t i = start; i < end; i++) {
        float c = (bestAxis == 0) ? prims[i].centroid.x
                : (bestAxis == 1) ? prims[i].centroid.y
                                  : prims[i].centroid.z;
        int b = (int)(LBVH_NUM_BINS * ((c - cmin) / axisExtent));
        if (b >= LBVH_NUM_BINS) b = LBVH_NUM_BINS - 1;
        bins[b].count++;
        bins[b].weight += prims[i].weight;
        bins[b].bounds.expand(prims[i].bounds);
    }

    // Prefix / suffix accumulations skipping empty bins.
    AABB  prefixBounds[LBVH_NUM_BINS];
    AABB  suffixBounds[LBVH_NUM_BINS];
    float prefixWeight[LBVH_NUM_BINS] = {};
    float suffixWeight[LBVH_NUM_BINS] = {};
    uint32_t prefixCount[LBVH_NUM_BINS] = {};
    uint32_t suffixCount[LBVH_NUM_BINS] = {};

    AABB  rLeft; float wLeft = 0.0f; uint32_t cLeft = 0;
    for (int i = 0; i < LBVH_NUM_BINS; i++) {
        if (bins[i].count > 0) {
            rLeft.expand(bins[i].bounds);
            wLeft += bins[i].weight;
            cLeft += bins[i].count;
        }
        prefixBounds[i] = rLeft;
        prefixWeight[i] = wLeft;
        prefixCount[i]  = cLeft;
    }
    AABB  rRight; float wRight = 0.0f; uint32_t cRight = 0;
    for (int i = LBVH_NUM_BINS - 1; i >= 0; i--) {
        if (bins[i].count > 0) {
            rRight.expand(bins[i].bounds);
            wRight += bins[i].weight;
            cRight += bins[i].count;
        }
        suffixBounds[i] = rRight;
        suffixWeight[i] = wRight;
        suffixCount[i]  = cRight;
    }

    // SAH-like cost with weight. Using weight * surfaceArea rather than
    // count * surfaceArea biases the split to keep high-power lights together
    // in tighter bounds, which improves the stochastic-descent heuristic at
    // traversal time (children whose weight dominates get picked more often,
    // and tighter bounds mean their distance estimate is more accurate).
    float bestCost  = 1e30f;
    int   bestSplit = -1;
    float totalSA   = totalBounds.surfaceArea();
    for (int s = 0; s < LBVH_NUM_BINS - 1; s++) {
        uint32_t cL = prefixCount[s];
        uint32_t cR = suffixCount[s + 1];
        if (cL == 0 || cR == 0) continue;
        if (totalSA < 1e-20f) continue;
        float wL = prefixWeight[s];
        float wR = suffixWeight[s + 1];
        float saL = prefixBounds[s].surfaceArea();
        float saR = suffixBounds[s + 1].surfaceArea();
        // Prefer weighted SAH when any light carries weight; fall back to count
        // when the whole subtree has zero weight (shouldn't happen, but safe).
        float cost;
        if (totalWeight > 0.0f) {
            cost = 0.125f + (wL * saL + wR * saR) / (totalWeight * totalSA);
        } else {
            cost = 0.125f + (cL * saL + cR * saR) / totalSA;
        }
        if (cost < bestCost) { bestCost = cost; bestSplit = s; }
    }

    if (bestSplit == -1) return makeLeaf();

    // Partition in place.
    uint32_t mid = start;
    for (uint32_t i = start; i < end; i++) {
        float c = (bestAxis == 0) ? prims[i].centroid.x
                : (bestAxis == 1) ? prims[i].centroid.y
                                  : prims[i].centroid.z;
        int b = (int)(LBVH_NUM_BINS * ((c - cmin) / axisExtent));
        if (b >= LBVH_NUM_BINS) b = LBVH_NUM_BINS - 1;
        if (b <= bestSplit) {
            std::swap(prims[i], prims[mid]);
            mid++;
        }
    }
    if (mid == start || mid == end) mid = (start + end) / 2;

    uint32_t nodeIdx = (uint32_t)nodes.size();
    LightBVHNode interior;
    memset(&interior, 0, sizeof(LightBVHNode));
    nodes.push_back(interior);

    uint32_t leftIdx  = buildRecursive(nodes, prims, orderedPrims, start, mid);
    uint32_t rightIdx = buildRecursive(nodes, prims, orderedPrims, mid, end);

    nodes[nodeIdx].bounds     = totalBounds;
    nodes[nodeIdx].weight     = totalWeight;
    nodes[nodeIdx].leftChild  = leftIdx;
    nodes[nodeIdx].rightChild = rightIdx;
    nodes[nodeIdx].primCount  = 0;
    return nodeIdx;
}

LightBVHData LightBVH::build(const AABB* bounds, const float* weights, uint32_t lightCount) {
    LightBVHData result;
    if (lightCount == 0) return result;

    std::vector<LBPrim> prims(lightCount);
    for (uint32_t i = 0; i < lightCount; i++) {
        prims[i].bounds    = bounds[i];
        prims[i].centroid  = bounds[i].center();
        prims[i].weight    = weights[i];
        prims[i].origIndex = i;
    }

    result.rootIndex = buildRecursive(result.nodes, prims, result.orderedLightIndices,
                                      0, lightCount);
    LOG_DEBUG("Light BVH: %u nodes, %u lights, root=%u",
              (uint32_t)result.nodes.size(), lightCount, result.rootIndex);
    return result;
}
