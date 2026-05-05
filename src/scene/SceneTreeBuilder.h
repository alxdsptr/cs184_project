#pragma once
#include "scene/Animation.h"
#include <assimp/scene.h>
#include <unordered_map>
#include <vector>

class Scene;

// Walk the Assimp node hierarchy and produce a parent-before-child ordered
// SceneNode array. Resolve Assimp's `$AssimpFbx$_*` pivot-chain decomposition
// so a single logical node carries the composed local transform that the
// pivot chain represents.
//
// The output:
//   - scene.getNodes() filled with parent-ordered SceneNode entries.
//   - aiNode -> SceneNode index map written into outNodeIndex (keyed by the
//     aiNode pointer; logical nodes inherit the index of their pivot-chain
//     leaf so animation channels can be resolved by name).
//   - outLogicalNameToNodeIndex: the *logical* node name (i.e. without the
//     `_$AssimpFbx$_*` suffix) → SceneNode index. Animation channels may
//     reference either the logical name or one of the pivot-chain
//     intermediates; both are inserted here. Channels targeting different
//     intermediates of the same logical node are summed by the caller into
//     a single composed transform (see SceneLoader::load animation parsing).
void buildSceneTree(
    const aiScene* aiScn,
    Scene& scene,
    std::unordered_map<const aiNode*, int>& outNodeIndex,
    std::unordered_map<std::string, int>& outLogicalNameToNodeIndex,
    std::unordered_map<std::string, int>& outAnyNameToNodeIndex);

// Strips the `_$AssimpFbx$_*` suffix from a node name. Returns the unmodified
// name when the marker is absent.
std::string stripAssimpPivotSuffix(const std::string& name);
