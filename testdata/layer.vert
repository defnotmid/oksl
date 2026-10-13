#version 300 es
// The layer vertex shader (shaders/layer.slang) in GLSL: a quad in target
// pixels (y down) with a uv per vertex, which the layer profiles' fragment
// shaders read at location 0. `odin run tools/gpu_parity -- -oksl-vertex`
// draws every testdata shader with it instead of the committed one.
precision highp float;

layout(location = 0) in vec2 position;
layout(location = 1) in vec2 uv;
layout(location = 0) out vec2 vUv;

// The target's size in pixels (xy), at offset 0 of the vertex block (set 1).
uniform vec4 size;

void main()
{
    vUv = uv;
    // SDL_GPU puts NDC +1 at the top row of every target.
    gl_Position = vec4(2.0 * position.x / size.x - 1.0, 1.0 - 2.0 * position.y / size.y, 0.0, 1.0);
}
