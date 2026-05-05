#include "app/Application.h"
#include "core/Types.h"
#include "scene/SceneLoader.h"
#include "util/Log.h"

#include <cstdint>
#include <cstdlib>
#include <string>

int main(int argc, char** argv) {
    std::string scenePath;
    std::string outputPath;
    std::string envMapPath;
    std::string cameraPath;
    uint32_t width = 1280;
    uint32_t height = 720;
    uint32_t maxBounces = 8;
    uint32_t samples = 1;
    uint32_t samplesPerFrame = 1;  // spp per realtime frame (independent of `-s`)
    int initialMode = -1;  // -1=default, 0=Native, 1=NRDOnly, 2=NRDDLSS, 3=DLSSOnly, 4=DLSSRR
    int backendKind = 0;   // 0=CUDA, 1=OptiX
    SGWorkflowMode sgMode = SGWorkflowMode::Off;
    float emissiveTargetLum = 20.0f;
    bool restirDI = true;
    bool restirGI = false;
    bool restirPT = false;
    bool mediumEnabled = false;
    float3 mediumSigmaA = make_float3(0.0f, 0.0f, 0.0f);
    float3 mediumSigmaS = make_float3(0.0f, 0.0f, 0.0f);
    float mediumDensity = 1.0f;
    float mediumAnisotropy = 0.0f;
    uint32_t mediumDensityKind = 0;  // 0=Constant, 1=HeightFalloff, 2=FBM, 3=HeightFBM
    bool mediumAnyOverride = false;
    bool mediumKindOverride = false;

    // Capture-mode CLI state. Setting --capture-tag activates it.
    Application::CaptureOptions capOpts;
    bool captureEnabled = false;

    // Replay-mode CLI state. Setting --replay activates it.
    Application::ReplayOptions replayOpts;
    bool replayEnabled = false;

    // Animation playback. --play-anim enables time-driven AnimationClip
    // sampling per render frame. Defaults to 30 fps (the FBX clip's tps).
    bool playAnim = false;
    float animFps = 30.0f;
    float animStartTime = 0.0f;

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
            else if (m == "dlssonly") initialMode = 3;
            else if (m == "rr" || m == "dlssrr") initialMode = 4;
            else LOG_WARN("Invalid --mode value: %s (use native|nrd|dlss|dlssonly|rr)", m.c_str());
        } else if (arg == "--backend" && i + 1 < argc) {
            std::string b = argv[++i];
            if (b == "cuda") backendKind = 0;
            else if (b == "optix") backendKind = 1;
            else LOG_WARN("Invalid --backend value: %s (use cuda|optix)", b.c_str());
        } else if (arg == "--sg" && i + 1 < argc) {
            std::string s = argv[++i];
            if (s == "off")           sgMode = SGWorkflowMode::Off;
            else if (s == "heuristic") sgMode = SGWorkflowMode::SpecLumHeuristic;
            else if (s == "fbx-c4d")  sgMode = SGWorkflowMode::FbxC4D;
            else if (s == "fbx-ue")   sgMode = SGWorkflowMode::FbxUE;
            else LOG_WARN("Invalid --sg value: %s (use off|heuristic|fbx-c4d|fbx-ue)", s.c_str());
        } else if (arg == "--no-restir") {
            restirDI = false;
        } else if (arg == "--restir-gi") {
            restirGI = true;
        } else if (arg == "--no-restir-gi") {
            restirGI = false;
        } else if (arg == "--restir-pt") {
            restirPT = true;
        } else if (arg == "--no-restir-pt") {
            restirPT = false;
        } else if (arg == "--emissive-target" && i + 1 < argc) {
            float v = (float)std::atof(argv[++i]);
            if (v > 0.0f) {
                emissiveTargetLum = v;
            } else {
                LOG_WARN("Invalid --emissive-target value: %s", argv[i]);
            }
        } else if (arg == "--medium" && i + 1 < argc) {
            std::string v = argv[++i];
            if (v == "on" || v == "1" || v == "true") {
                mediumEnabled = true;
                mediumAnyOverride = true;
            } else if (v == "off" || v == "0" || v == "false") {
                mediumEnabled = false;
                mediumAnyOverride = true;
            } else {
                LOG_WARN("Invalid --medium value: %s (use on|off)", v.c_str());
            }
        } else if (arg == "--sigma-a" && i + 3 < argc) {
            // Read sequentially — argument-evaluation order in a single
            // function call is unspecified in C++, so MSVC could (and does)
            // grab args right-to-left and silently transpose RGB→BGR.
            float r = (float)std::atof(argv[++i]);
            float g = (float)std::atof(argv[++i]);
            float b = (float)std::atof(argv[++i]);
            mediumSigmaA = make_float3(r, g, b);
            mediumAnyOverride = true;
        } else if (arg == "--sigma-s" && i + 3 < argc) {
            float r = (float)std::atof(argv[++i]);
            float g = (float)std::atof(argv[++i]);
            float b = (float)std::atof(argv[++i]);
            mediumSigmaS = make_float3(r, g, b);
            mediumAnyOverride = true;
        } else if (arg == "--medium-density" && i + 1 < argc) {
            float v = (float)std::atof(argv[++i]);
            if (v >= 0.0f) {
                mediumDensity = v;
                mediumAnyOverride = true;
            } else {
                LOG_WARN("Invalid --medium-density value: %s", argv[i]);
            }
        } else if (arg == "--medium-g" && i + 1 < argc) {
            float v = (float)std::atof(argv[++i]);
            if (v >= -0.99f && v <= 0.99f) {
                mediumAnisotropy = v;
                mediumAnyOverride = true;
            } else {
                LOG_WARN("Invalid --medium-g value: %s (range -0.99..0.99)", argv[i]);
            }
        } else if (arg == "--medium-kind" && i + 1 < argc) {
            std::string v = argv[++i];
            if      (v == "constant" || v == "homogeneous") mediumDensityKind = 0;
            else if (v == "height" || v == "height-falloff") mediumDensityKind = 1;
            else if (v == "fbm" || v == "noise")             mediumDensityKind = 2;
            else if (v == "height-fbm" || v == "smoke")      mediumDensityKind = 3;
            else { LOG_WARN("Invalid --medium-kind value: %s (constant|height|fbm|height-fbm)", v.c_str()); continue; }
            mediumKindOverride = true;
            mediumAnyOverride = true;
        } else if ((arg == "--spp" || arg == "-p") && i + 1 < argc) {
            int value = std::atoi(argv[++i]);
            if (value > 0) {
                samplesPerFrame = (uint32_t)value;
            } else {
                LOG_WARN("Invalid spp value: %s", argv[i]);
            }
        } else if (arg == "-f" && i + 1 < argc) {
            outputPath = argv[++i];
        } else if (arg == "--camera" && i + 1 < argc) {
            cameraPath = argv[++i];
        } else if (arg == "--capture-tag" && i + 1 < argc) {
            capOpts.tag = argv[++i];
            captureEnabled = true;
        } else if (arg == "--capture-out" && i + 1 < argc) {
            capOpts.outDir = argv[++i];
        } else if (arg == "--capture-warmup" && i + 1 < argc) {
            int v = std::atoi(argv[++i]);
            if (v >= 0) capOpts.warmupFrames = (uint32_t)v;
        } else if (arg == "--capture-frames" && i + 1 < argc) {
            int v = std::atoi(argv[++i]);
            if (v > 0) capOpts.captureFrames = (uint32_t)v;
        } else if (arg == "--capture-stride" && i + 1 < argc) {
            int v = std::atoi(argv[++i]);
            if (v > 0) capOpts.captureStride = (uint32_t)v;
        } else if (arg == "--capture-dwell" && i + 1 < argc) {
            // Frames to hold the camera still at each capture point before
            // saving. Default 1 = original behavior. Use a large value (e.g.
            // 2000) for a "near-converged reference" sweep that shares the
            // same camera path as the test sweeps.
            int v = std::atoi(argv[++i]);
            if (v > 0) capOpts.dwellFrames = (uint32_t)v;
        } else if (arg == "--capture-fps" && i + 1 < argc) {
            // Virtual frame rate driving the camera path during the motion
            // phase. Decoupled from the real render fps, so frame N is at
            // the same camera pose across all ReSTIR modes regardless of
            // their actual throughput. Default 60.
            float v = (float)std::atof(argv[++i]);
            if (v > 0.0f) capOpts.fixedStepFps = v;
        // Motion-mode selector: --capture-orbit switches from the default
        // forward-dolly to a circular orbit. Mutually exclusive with the
        // dolly knobs (which are simply ignored when orbit is selected).
        } else if (arg == "--capture-orbit") {
            capOpts.motion = Application::CaptureMotion::Orbit;
        } else if (arg == "--capture-dolly") {
            // Explicit form for clarity; same as the default.
            capOpts.motion = Application::CaptureMotion::Dolly;
        } else if (arg == "--capture-speed" && i + 1 < argc) {
            // Dolly speed in world units per second. Default 0.3 — slow
            // enough that a 5-second capture moves the camera ~1.5 units,
            // useful for medium-scale scenes (Sponza ~30 units across).
            // Allow 0 explicitly: static camera is a useful capture mode
            // for "render N spp at one fixed pose" workflows (reference
            // sweep in run_quality_sweep.sh).
            float v = (float)std::atof(argv[++i]);
            if (v >= 0.0f) capOpts.dollySpeed = v;
        } else if (arg == "--capture-period" && i + 1 < argc) {
            // Orbit period (seconds per revolution). Only used in orbit mode.
            float v = (float)std::atof(argv[++i]);
            if (v > 0.0f) capOpts.orbitPeriodSeconds = v;
        } else if (arg == "--capture-radius" && i + 1 < argc) {
            float v = (float)std::atof(argv[++i]);
            if (v > 0.0f) capOpts.orbitRadius = v;
        } else if (arg == "--capture-pitch" && i + 1 < argc) {
            capOpts.orbitPitchDeg = (float)std::atof(argv[++i]);
        } else if (arg == "--replay" && i + 1 < argc) {
            replayOpts.recordingPath = argv[++i];
            replayEnabled = true;
        } else if (arg == "--replay-out" && i + 1 < argc) {
            replayOpts.outDir = argv[++i];
        } else if (arg == "--replay-spp" && i + 1 < argc) {
            int v = std::atoi(argv[++i]);
            if (v > 0) replayOpts.sppPerPose = (uint32_t)v;
        } else if (arg == "--replay-stride" && i + 1 < argc) {
            int v = std::atoi(argv[++i]);
            if (v > 0) replayOpts.stride = (uint32_t)v;
        } else if (arg == "--replay-max" && i + 1 < argc) {
            int v = std::atoi(argv[++i]);
            if (v >= 0) replayOpts.maxPoses = (uint32_t)v;
        } else if (arg == "--play-anim") {
            // Enable time-driven AnimationClip playback. The renderer
            // advances animation by 1/animFps each rendered frame.
            playAnim = true;
        } else if (arg == "--anim-fps" && i + 1 < argc) {
            float v = (float)std::atof(argv[++i]);
            if (v > 0.0f) animFps = v; else LOG_WARN("Invalid --anim-fps: %s", argv[i]);
        } else if (arg == "--anim-start" && i + 1 < argc) {
            float v = (float)std::atof(argv[++i]);
            animStartTime = v;
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
    app.setBackendKind(backendKind);
    app.setSGWorkflowMode(sgMode);
    app.setEmissiveTargetLum(emissiveTargetLum);
    app.setReSTIREnabled(restirDI);
    app.setReSTIRGIEnabled(restirGI);
    app.setReSTIRPTEnabled(restirPT);
    app.setPlayAnimation(playAnim);
    app.setAnimationFps(animFps);
    app.setAnimationStartTime(animStartTime);
    if (mediumAnyOverride) {
        app.setMediumEnabled(mediumEnabled);
        app.setMediumSigmaA(mediumSigmaA);
        app.setMediumSigmaS(mediumSigmaS);
        app.setMediumDensity(mediumDensity);
        app.setMediumAnisotropy(mediumAnisotropy);
        if (mediumKindOverride) app.setMediumDensityKind(mediumDensityKind);
    }
    if (!outputPath.empty()) {
        app.setHeadlessOutput(outputPath, samples);
    }
    if (!cameraPath.empty()) {
        app.loadCameraFile(cameraPath);
    }
    if (captureEnabled) {
        app.setCaptureOptions(capOpts);
    }
    if (replayEnabled) {
        app.setReplayOptions(replayOpts);
    }

    // Replay mode runs without GUI (loops poses internally and exits).
    bool enableGui = outputPath.empty() && !replayEnabled;
    if (!app.init(width, height, "CUDA Path Tracer", enableGui)) {
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
