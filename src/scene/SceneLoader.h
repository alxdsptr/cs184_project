#pragma once
#include "scene/Scene.h"
#include <string>

// How to interpret the legacy Specular-Glossiness workflow when loading FBX/etc.
// assets. The SG path needs per-asset knowledge because different DCC tools
// pack the *_Specular.dds map differently.
enum class SGWorkflowMode {
    Off,        // Disable SG entirely; surfaces fall back to MR defaults
                // (white albedo + roughness=0.045 baseline). Default — safe
                // for any asset whose Specular map packing we don't recognise.
    SpecLumHeuristic,
                // Enable SG with the spec-luminance fallback: ignore RGB
                // semantics, drive F0 and roughness from sqrt(spec luminance).
                // A blunt instrument that rarely matches an artist's intent;
                // kept around as an escape hatch for unknown FBX packings.
    FbxC4D,     // Enable SG + C4D custom packing (B → spec strength,
                // G → roughness). Use for MEASURE_SEVEN-style assets.
    FbxUE       // Enable SG + UE/standard PBR-Specular packing
                // (G → glossiness, B → metallic mask). Use for Bistro and
                // most NVIDIA-distributed FBX assets.
};

class SceneLoader {
public:
    static bool load(const std::string& path,
                     Scene& scene,
                     SGWorkflowMode sgMode = SGWorkflowMode::Off,
                     float texturedEmissiveTargetLum = 20.0f);
};
