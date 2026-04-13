#pragma once

static const char* g_quadVertSrc = R"glsl(
#version 330 core
out vec2 vUV;
void main() {
    // Fullscreen triangle trick: 3 vertices cover the screen
    float x = float((gl_VertexID & 1) << 2) - 1.0;
    float y = float((gl_VertexID & 2) << 1) - 1.0;
    // OpenGL texture origin is bottom-left; flip V to match top-left render buffer rows.
    vUV = vec2((x + 1.0) * 0.5, 1.0 - (y + 1.0) * 0.5);
    gl_Position = vec4(x, y, 0.0, 1.0);
}
)glsl";

static const char* g_quadFragSrc = R"glsl(
#version 330 core
in vec2 vUV;
out vec4 fragColor;
uniform sampler2D uTex;
void main() {
    fragColor = texture(uTex, vUV);
}
)glsl";
