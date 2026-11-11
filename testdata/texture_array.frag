#version 300 es
precision highp float;
// A sampler2DArray (tools/gpu_parity's three layers): the layer from uv,
// at the level texture() picks, at an explicit one, and at the level
// textureGrad's derivatives pick, on an array and a 2D texture.
uniform sampler2DArray layers;
uniform sampler2D plain;
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;
void main() {
    vec2 st = uv * vec2(3.0, 2.0);
    float layer = floor(uv.x * 3.0);
    vec4 a = texture(layers, vec3(st, layer));
    vec4 b = textureLod(layers, vec3(st * 0.5, 2.0 - layer), 1.5);
    vec4 c = textureGrad(plain, st, dFdx(st) * 3.0, dFdy(st));
    vec4 d = textureGrad(layers, vec3(st, 1.0), dFdx(st), dFdy(st) * 2.0);
    ivec3 size = textureSize(layers, 0);
    color = (mix(mix(a, b, 0.5), mix(c, d, 0.5), 0.5)) * float(size.z) / 3.0;
}
