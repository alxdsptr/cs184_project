#include "scene/SceneTreeBuilder.h"
#include "scene/Scene.h"
#include "core/Math.h"
#include "util/Log.h"

#include <cstring>

static float4x4 toFloat4x4(const aiMatrix4x4& m) {
    float4x4 r{};
    r.m[0][0] = m.a1; r.m[0][1] = m.a2; r.m[0][2] = m.a3; r.m[0][3] = m.a4;
    r.m[1][0] = m.b1; r.m[1][1] = m.b2; r.m[1][2] = m.b3; r.m[1][3] = m.b4;
    r.m[2][0] = m.c1; r.m[2][1] = m.c2; r.m[2][2] = m.c3; r.m[2][3] = m.c4;
    r.m[3][0] = m.d1; r.m[3][1] = m.d2; r.m[3][2] = m.d3; r.m[3][3] = m.d4;
    return r;
}

std::string stripAssimpPivotSuffix(const std::string& name) {
    static const char* MARKER = "_$AssimpFbx$_";
    size_t pos = name.find(MARKER);
    if (pos == std::string::npos) return name;
    return name.substr(0, pos);
}

// Recursive builder. We do NOT collapse pivot chains: every aiNode becomes its
// own SceneNode (even the `_$AssimpFbx$_*` intermediates), so animation
// channels keyed on intermediate names land on the *exact* node whose
// localRest should be replaced. Folding them into a logical leaf would lose
// the per-intermediate decomposition (Translation only carries T, Rotation
// only carries R, etc., so a leaf-overriding evaluator would have to know
// which TRS slots to leave alone — much messier than just keeping the chain).
//
// The cost is a few thousand extra SceneNodes for FBX scenes — trivial in
// memory, and the per-frame hierarchy walk is one mat-mat per node.
static void buildRecursive(
    const aiNode* node,
    int parentSceneIdx,
    Scene& scene,
    std::unordered_map<const aiNode*, int>& nodeIndex,
    std::unordered_map<std::string, int>& logicalNameToIdx,
    std::unordered_map<std::string, int>& anyNameToIdx)
{
    auto& nodes = scene.getNodes();
    int myIdx = (int)nodes.size();
    nodes.emplace_back();
    SceneNode& sn = nodes.back();

    sn.name      = node->mName.C_Str();
    sn.parent    = parentSceneIdx;
    sn.localRest = toFloat4x4(node->mTransformation);

    if (parentSceneIdx >= 0) {
        sn.worldRest = mat4_multiply(nodes[parentSceneIdx].worldRest, sn.localRest);
    } else {
        sn.worldRest = sn.localRest;
    }
    sn.meshCount = node->mNumMeshes;

    nodeIndex[node] = myIdx;
    anyNameToIdx[sn.name] = myIdx;
    // Logical-name index: strip `_$AssimpFbx$_*` suffix. For non-pivot nodes
    // this is just the bare name. Multiple pivot intermediates map to the
    // same logical name; we keep the FIRST encountered in the map (later
    // overrides would be misleading).
    std::string logical = stripAssimpPivotSuffix(sn.name);
    auto inserted = logicalNameToIdx.emplace(logical, myIdx);
    (void)inserted;

    for (unsigned i = 0; i < node->mNumChildren; i++) {
        buildRecursive(node->mChildren[i], myIdx,
                       scene, nodeIndex, logicalNameToIdx, anyNameToIdx);
    }
}

void buildSceneTree(
    const aiScene* aiScn,
    Scene& scene,
    std::unordered_map<const aiNode*, int>& outNodeIndex,
    std::unordered_map<std::string, int>& outLogicalNameToNodeIndex,
    std::unordered_map<std::string, int>& outAnyNameToNodeIndex)
{
    auto& nodes = scene.getNodes();
    nodes.clear();
    outNodeIndex.clear();
    outLogicalNameToNodeIndex.clear();
    outAnyNameToNodeIndex.clear();

    if (!aiScn || !aiScn->mRootNode) return;

    nodes.reserve(16384);
    buildRecursive(aiScn->mRootNode, -1, scene,
                   outNodeIndex, outLogicalNameToNodeIndex, outAnyNameToNodeIndex);

    LOG_INFO("Built SceneNode tree: %zu nodes (1:1 with aiNodes)", nodes.size());
}
