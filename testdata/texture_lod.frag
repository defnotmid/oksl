#version 300 es
precision highp float;

// Mip levels in a GLSL fragment shader, in four bands: texture() at the
// level its derivatives pick, the same with a bias, textureLod, and
// texelFetch of level 1 sized by textureSize. Every sample is taken
// outside the branches, where the derivatives are defined.
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;

uniform sampler2D tex;

void main()
{
    // 1 to 40 repeats across: magnified at the left, down to the last
    // levels at the right; the rows bend, so x and y shrink differently.
    float repeats = mix(1.0, 40.0, uv.x * uv.x);
    vec2 st = vec2(uv.x * repeats, uv.y * 2.0 + sin(uv.x * 6.0) * 0.3);
    vec4 implicit = texture(tex, st);
    vec4 biased = texture(tex, st, 1.5);
    vec4 explicit = textureLod(tex, st, uv.x * 6.0);
    ivec2 size = textureSize(tex, 1);
    vec4 fetched = texelFetch(tex, ivec2(gl_FragCoord.xy / 4.0) % size, 1);
    float band = floor(uv.y * 4.0);
    color = band == 0.0 ? implicit : band == 1.0 ? biased : band == 2.0 ? explicit : fetched;
    color.a = 1.0;
}
