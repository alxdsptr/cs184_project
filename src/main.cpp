#include "app/Application.h"
#include "util/Log.h"

#include <cstdint>
#include <cstdlib>
#include <string>

int main(int argc, char** argv) {
    std::string scenePath;
    std::string outputPath;
    std::string envMapPath;
    uint32_t width = 1280;
    uint32_t height = 720;
    uint32_t maxBounces = 8;
    uint32_t samples = 1;
    uint32_t samplesPerFrame = 1;  // spp per realtime frame (independent of `-s`)
    int initialMode = -1;  // -1=default, 0=Native, 1=NRDOnly, 2=NRDDLSS

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-e" && i + 1 < argc) {
            envMapPath = argv[++i];
        } else if (arg == "-m" && i + 1 < argc) {
            int value = std::atoi(argv[++i]);
            if (value > 0) {
                maxBounces = (uint32_t)value;
            } else {
                LOG_WARN("Invalid max bounce value: %s", argv[i]);
            }
        } else if (arg == "-s" && i + 1 < argc) {
            int value = std::atoi(argv[++i]);
            if (value > 0) {
                samples = (uint32_t)value;
            } else {
                LOG_WARN("Invalid sample count value: %s", argv[i]);
            }
        } else if (arg == "--mode" && i + 1 < argc) {
            std::string m = argv[++i];
            if (m == "native") initialMode = 0;
            else if (m == "nrd" || m == "nrdonly") initialMode = 1;
            else if (m == "dlss" || m == "nrddlss") initialMode = 2;
            else LOG_WARN("Invalid --mode value: %s (use native|nrd|dlss)", m.c_str());
        } else if ((arg == "--spp" || arg == "-p") && i + 1 < argc) {
            int value = std::atoi(argv[++i]);
            if (value > 0) {
                samplesPerFrame = (uint32_t)value;
            } else {
                LOG_WARN("Invalid spp value: %s", argv[i]);
            }
        } else if (arg == "-f" && i + 1 < argc) {
            outputPath = argv[++i];
        } else if (arg == "-r" && i + 2 < argc) {
            int parsedWidth = std::atoi(argv[++i]);
            int parsedHeight = std::atoi(argv[++i]);
            if (parsedWidth > 0 && parsedHeight > 0) {
                width = (uint32_t)parsedWidth;
                height = (uint32_t)parsedHeight;
            } else {
                LOG_WARN("Invalid resolution value: %s x %s", argv[i - 1], argv[i]);
            }
        } else if (!arg.empty() && arg[0] != '-') {
            if (scenePath.empty()) {
                scenePath = arg;
            } else {
                LOG_WARN("Ignoring extra positional argument: %s", arg.c_str());
            }
        } else {
            LOG_WARN("Ignoring unknown argument: %s", arg.c_str());
        }
    }

    if (!outputPath.empty() && scenePath.empty()) {
        LOG_ERROR("-f requires a scene file argument");
        return 1;
    }

    Application app;
    app.setMaxBounces(maxBounces);
    app.setSamplesPerFrame(samplesPerFrame);
    app.setInitialMode(initialMode);
    if (!outputPath.empty()) {
        app.setHeadlessOutput(outputPath, samples);
    }

    if (!app.init(width, height, "CUDA Path Tracer", outputPath.empty())) {
        return 1;
    }

    if (!scenePath.empty()) {
        if (!app.loadScene(scenePath)) {
            LOG_ERROR("Usage: pathtracer <scene_file>");
        }
    } else {
        LOG_INFO("No scene file specified. Pass a glTF/OBJ file as argument.");
        LOG_INFO("Example: pathtracer assets/DamagedHelmet.glb");
    }

    if (!envMapPath.empty()) {
        app.setEnvMap(envMapPath);
    }

    app.run();
    app.shutdown();
    return 0;
}
