#pragma once
#include "scene/Scene.h"
#include <string>

class SceneLoader {
public:
    static bool load(const std::string& path, Scene& scene);
};
