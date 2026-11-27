// pow with literal whole exponents 3 to 5 over negative bases: n - 1
// multiplies on every backend (exp2(n log2 x) is NaN on WARP), as a cube,
// a vector fourth power and Schlick's Fresnel term pow(1.0 - c, 5.0) that
// o3d shades with, c past 1 here.
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 p = (2.0 * fragCoord - iResolution.xy) / iResolution.y;
    float c = pow(p.x, 3.0) * 0.5 + 0.5;
    vec2 q = pow(p - 0.25, vec2(4.0));
    float f = 0.04 + 0.96 * pow(1.0 - length(p), 5.0);
    fragColor = vec4(c, q.y, f * 0.5 + 0.5, 1.0);
}
