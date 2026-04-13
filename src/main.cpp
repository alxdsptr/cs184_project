#include "app/Application.h"
#include "util/Log.h"

int main(int argc, char** argv) {
    Application app;
    if (!app.init(1280, 720, "CUDA Path Tracer")) {
        return 1;
    }

    if (argc > 1) {
        if (!app.loadScene(argv[1])) {
            LOG_ERROR("Usage: pathtracer <scene_file>");
        }
    } else {
        LOG_INFO("No scene file specified. Pass a glTF/OBJ file as argument.");
        LOG_INFO("Example: pathtracer assets/DamagedHelmet.glb");
    }

    app.run();
    app.shutdown();
    return 0;
}
