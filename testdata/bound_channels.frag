#version 310 es
precision highp float;

// Single-level float channel images (r32f, rg32f and rgba32f) with nearest
// (channels 0..2) and bilinear (channels 3..5) edge-clamped sampling.
// Top half: nearest at texel centres, clamped out-of-range UVs, texelFetch
// and textureSize (exact match with GPU). Bottom half: bilinear across
// interior and edge-clamped UVs, non-zero lod and textureGrad.
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;

uniform sampler2D sRNear;
uniform sampler2D sRgNear;
uniform sampler2D sRgbaNear;
uniform sampler2D sRLin;
uniform sampler2D sRgLin;
uniform sampler2D sRgbaLin;

void main()
{
    ivec2 px = ivec2(gl_FragCoord.xy);
    ivec2 szR = max(textureSize(sRNear, 0), ivec2(1));
    ivec2 szRg = max(textureSize(sRgNear, 0), ivec2(1));
    ivec2 szRgba = max(textureSize(sRgbaNear, 0), ivec2(1));

    ivec2 pR = ivec2(px.x % szR.x, px.y % szR.y);
    ivec2 pRg = ivec2(px.x % szRg.x, px.y % szRg.y);
    ivec2 pRgba = ivec2(px.x % szRgba.x, px.y % szRgba.y);

    vec2 cR = (vec2(pR) + 0.5) / vec2(szR);
    vec2 cRg = (vec2(pRg) + 0.5) / vec2(szRg);
    vec2 cRgba = (vec2(pRgba) + 0.5) / vec2(szRgba);

    float edgeU = (px.x & 2) == 0 ? -0.75 : 1.75;
    float edgeV = (px.y & 2) == 0 ? -0.5 : 1.5;
    vec2 clampUv = (px.x & 4) == 0 ? vec2(edgeU, cRgba.y) : vec2(cRgba.x, edgeV);

    vec4 n0 = textureLod(sRNear, cR, 0.0);
    vec4 n1 = textureLod(sRgNear, cRg, 3.0);
    vec4 n2 = textureLod(sRgbaNear, cRgba, 0.0);
    vec4 n3 = texture(sRgbaNear, clampUv);
    vec4 n4 = texelFetch(sRNear, pR, 0);
    vec4 n5 = texelFetch(sRgNear, pRg, 0);
    vec4 n6 = texelFetch(sRgbaNear, pRgba, 0);
    vec4 n7 = vec4(vec2(szR), vec2(szRg)) - vec4(vec2(szRgba), 0.0, 0.0);

    vec2 localUv = vec2(fract(uv.x * 4.0), (uv.y - 0.5) * 2.0);
    vec2 linUv = localUv * 1.5 - 0.25;
    vec4 l0 = textureLod(sRLin, linUv, 0.0);
    vec4 l1 = textureLod(sRgLin, linUv, 1.5);
    vec4 l2 = textureLod(sRgbaLin, linUv, 0.0);
    vec4 l3 = textureGrad(sRgbaLin, linUv, vec2(0.08, 0.0), vec2(0.0, 0.08));

    int band = clamp(px.x / 32, 0, 7);
    if (px.y < 72) {
        color = band == 0 ? n0 : band == 1 ? n1 : band == 2 ? n2 : band == 3 ? n3 :
                band == 4 ? n4 : band == 5 ? n5 : band == 6 ? n6 : n7;
    } else {
        int lband = band / 2;
        color = lband == 0 ? l0 : lband == 1 ? l1 : lband == 2 ? l2 : l3;
    }
}
